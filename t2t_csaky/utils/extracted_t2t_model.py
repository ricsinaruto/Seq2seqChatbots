from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import copy
import functools
import math
import time
import six
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.problem import problem_hparams_to_features
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities  # pylint: disable=unused-import
from tensor2tensor.utils import decoding
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils import metrics
from tensor2tensor.utils import optimize
from tensor2tensor.utils import quantization
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

from tensor2tensor.utils import beam_search

import tensorflow as tf

from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope

class ExtractedT2TModel(t2t_model.T2TModel):

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               data_parallelism=None,
               decode_hparams=None):
    super(ExtractedT2TModel, self).__init__(hparams,
                                            mode,
                                            problem_hparams,
                                            data_parallelism,
                                            decode_hparams)

  def call(self, inputs, **kwargs):
    del kwargs
    features = inputs
    t2t_model.set_custom_getter_compose(self._custom_getter)
    tf.get_variable_scope().set_initializer(
        optimize.get_variable_initializer(self.hparams))
    with self._eager_var_store.as_default():
      self._fill_problem_hparams_features(features)
      sharded_features = self._shard_features(features)
      sharded_logits, losses, \
      sharded_enc_ou = self.model_fn_sharded(sharded_features)
      if isinstance(sharded_logits, dict):
        concat_enc_ou = {}
        for k, v in six.iteritems(sharded_enc_ou):
          concat_enc_ou[k] = tf.concat(v, 0)
        return concat_enc_ou
      else:
        return tf.concat(sharded_enc_ou, 0)

  def model_fn_sharded(self, sharded_features):
    dp = self._data_parallelism
    t2t_model.summarize_features(sharded_features, num_shards=dp.n)
    datashard_to_features = self._to_features_per_datashard(sharded_features)
    sharded_logits, sharded_losses, enc_out \
      = dp(self.model_fn, datashard_to_features)
    if isinstance(sharded_logits[0], dict):
      temp_dict = {k: [] for k, _ in six.iteritems(sharded_logits[0])}
      for k, _ in six.iteritems(sharded_logits[0]):
        for l in sharded_logits:
          temp_dict[k].append(l[k])
      sharded_logits = temp_dict
    losses = t2t_model.average_sharded_losses(sharded_losses)

    return sharded_logits, losses, enc_out

  def model_fn(self, features):
    with tf.variable_scope(tf.get_variable_scope(), use_resource=True):
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16)

      with tf.variable_scope("body"):
        t2t_model.log_info("Building model body")
        body_out, enc_out = self.body(transformed_features)
      output, losses = self._normalize_body_output(body_out)

      if "training" in losses:
        t2t_model.log_info("Skipping T2TModel top and loss "
                           "because training loss "
                 "returned from body")
        logits = output
      else:
        logits = self.top(output, features)
        losses["training"] = 0.0
        if self._hparams.mode != tf.estimator.ModeKeys.PREDICT:
          losses["training"] = self.loss(logits, features)

      return logits, losses, enc_out

  def estimator_spec_predict(self, features, use_tpu=False):
    """Construct EstimatorSpec for PREDICT mode."""
    decode_hparams = self._decode_hparams
    infer_out = self.infer(
        features,
        beam_size=decode_hparams.beam_size,
        top_beams=(decode_hparams.beam_size
                   if decode_hparams.return_beams else 1),
        alpha=decode_hparams.alpha,
        decode_length=decode_hparams.extra_length,
        use_tpu=use_tpu)
    if isinstance(infer_out, dict):
      outputs = infer_out["outputs"]
      scores = infer_out["scores"]
      encoder_outputs = infer_out["encoder_outputs"]
    else:
      outputs = infer_out
      scores = None
      encoder_outputs = None

    inputs = features.get("inputs")
    if inputs is None:
      inputs = features["targets"]

    predictions = {
        "outputs": outputs,
        "scores": scores,
        "encoder_outputs": encoder_outputs,
        "inputs": inputs,
        "targets": features.get("infer_targets"),
        "batch_prediction_key": features.get("batch_prediction_key"),
    }
    t2t_model._del_dict_nones(predictions)

    export_out = {"outputs": predictions["outputs"]}
    if "scores" in predictions:
      export_out["scores"] = predictions["scores"]

    if "encoder_outputs" in predictions:
      export_out[]

    # Necessary to rejoin examples in the correct order with the Cloud ML Engine
    # batch prediction API.
    if "batch_prediction_key" in predictions:
      export_out["batch_prediction_key"] = predictions["batch_prediction_key"]

      t2t_model._remove_summaries()

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(export_out)
    }
    if use_tpu:
      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs=export_outputs)
    else:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs=export_outputs)

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """A inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: bool, whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
      if slow greedy decoding is used then the dict will also contain {
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`
      }
    """
    t2t_model.set_custom_getter_compose(self._custom_getter)
    with self._eager_var_store.as_default():
      # TODO(rsepassi): Make decoding work with real-valued model outputs
      # (i.e. if the target modality is RealModality).
      self.prepare_features_for_infer(features)
      if not self.has_input and beam_size > 1:
        t2t_model.log_warn("Beam searching for a model with no inputs.")
      if not self.has_input and self.hparams.sampling_method != "random":
        t2t_model.log_warn("Non-random sampling for a model with no inputs.")
      self._fill_problem_hparams_features(features)

      if self._problem_hparams:
        target_modality = self._problem_hparams.target_modality
        if target_modality.is_class_modality:
          beam_size = 1  # No use to run beam-search for a single class.
      t2t_model.log_info("Greedy Decoding")
      results = self._greedy_infer(features, decode_length, use_tpu)

      return results
