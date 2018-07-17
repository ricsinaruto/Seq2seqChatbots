
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tensor2tensor imports
from tensor2tensor.models import lstm
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import optimize

# tensorflow imports
import tensorflow as tf

import math
import six


# my imports
from t2t_csaky.hparams import seq2seq_hparams
from t2t_csaky.utils import optimizer
from t2t_csaky.models import gradient_checkpointed_seq2seq


@registry.register_model
class GradientCheckpointedSeq2seq(
    gradient_checkpointed_seq2seq.GradientCheckpointedSeq2seq):
  """
  This class is the modified version of the original Seq2Seq, where
  the outputs of the model are the encoded hidden state representations
  of the sentences.
  """

  REGISTERED_NAME = 'gradient_checkpointed_seq2seq'

  def __init__(self, *args, **kwargs):
    super(GradientCheckpointedSeq2seq, self).__init__(*args, **kwargs)
    self._name = 'gradient_checkpointed_seq2seq'

  def body(self, features):
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train=self._hparams.mode==tf.estimator.ModeKeys.TRAIN

    # Modified
    # using the custom lstm_seq2seq_internal_dynamic

    return gradient_checkpointed_seq2seq.lstm_seq2seq_internal_dynamic(
        features.get("inputs"),
        features["targets"],
        seq2seq_hparams.chatbot_lstm_hparams(),
        train)

  def call(self, inputs, **kwargs):
    del kwargs
    features = inputs
    t2t_model.set_custom_getter_compose(self._custom_getter)
    tf.get_variable_scope().set_initializer(
      optimize.get_variable_initializer(self.hparams))
    with self._eager_var_store.as_default():
      self._fill_problem_hparams_features(features)

      # Modified
      # passing the encoder state reference in 'sharded_enc_ou' variable

      sharded_features = self._shard_features(features)
      sharded_logits, losses, sharded_enc_ou \
        = self.model_fn_sharded(sharded_features)
      if isinstance(sharded_logits, dict):
        concat_logits = {}
        for k, v in six.iteritems(sharded_logits):
          concat_logits[k] = tf.concat(v, 0)
        concat_enc_ou = {}
        for k, v in six.iteritems(sharded_enc_ou):
          concat_enc_ou[k] = tf.concat(v, 0)

        return concat_logits, losses, concat_enc_ou
      else:
        return tf.concat(sharded_logits, 0), losses, sharded_enc_ou[0].h

  def model_fn_sharded(self, sharded_features):
    dp = self._data_parallelism
    t2t_model.summarize_features(sharded_features, num_shards=dp.n)
    datashard_to_features = self._to_features_per_datashard(sharded_features)

    # Modified
    # passing the encoder state reference in 'enc_out' variable

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

        # Modified
        # passing the encoder state reference in 'enc_out' variable

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

    # Modified
    # retrieving encoder output reference from inference output, and
    # adding it to the output predictions dictionary

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

    # Modified
    # Added encoder outputs to export outputs dictionary

    export_out = {"outputs": predictions["outputs"]}
    if "scores" in predictions:
      export_out["scores"] = predictions["scores"]

    if "encoder_outputs" in predictions:
      export_out["encoder_outputs"] = predictions["encoder_outputs"]

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

      # Modified
      # Removed every other decoding option, but the greedy method

      results = self._greedy_infer(features, decode_length, use_tpu)

      return results

  def _slow_greedy_infer(self, features, decode_length):
    """A slow greedy inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": None
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`}
      }
    """
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)
    if not self.has_input:
      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      features["partial_targets"] = tf.to_int64(partial_targets)
    # Save the targets in a var and reassign it after the tf.while loop to avoid
    # having targets being in a 'while' frame. This ensures targets when used
    # in metric functions stays in the same frame as other vars.
    targets_old = features.get("targets", None)

    target_modality = self._problem_hparams.target_modality

    def infer_step(recent_output, recent_logits, unused_loss):
      """Inference step."""
      if not tf.contrib.eager.in_eager_mode():
        if self._target_modality_is_real:
          dim = self._problem_hparams.target_modality.top_dimensionality
          recent_output.set_shape([None, None, None, dim])
        else:
          recent_output.set_shape([None, None, None, 1])
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      # This is inefficient in that it generates samples at all timesteps,
      # not just the last one, except if target_modality is pointwise.
      samples, logits, losses, enc_out = self.sample(features)
      # Concatenate the already-generated recent_output with last timestep
      # of the newly-generated samples.
      if target_modality.top_is_pointwise:
        cur_sample = samples[:, -1, :, :]
      else:
        cur_sample = samples[:,
                             common_layers.shape_list(recent_output)[1], :, :]
      if self._target_modality_is_real:
        cur_sample = tf.expand_dims(cur_sample, axis=1)
        samples = tf.concat([recent_output, cur_sample], axis=1)
      else:
        cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
        samples = tf.concat([recent_output, cur_sample], axis=1)
        if not tf.contrib.eager.in_eager_mode():
          samples.set_shape([None, None, None, 1])

      # Assuming we have one shard for logits.
      logits = tf.concat([recent_logits, logits[:, -1:]], 1)
      loss = sum([l for l in losses.values() if l is not None])
      return samples, logits, loss, enc_out

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if "partial_targets" in features:
      initial_output = tf.to_int64(features["partial_targets"])
      while len(initial_output.get_shape().as_list()) < 4:
        initial_output = tf.expand_dims(initial_output, 2)
      batch_size = common_layers.shape_list(initial_output)[0]
    else:
      batch_size = common_layers.shape_list(features["inputs"])[0]
      if self._target_modality_is_real:
        dim = self._problem_hparams.target_modality.top_dimensionality
        initial_output = tf.zeros((batch_size, 0, 1, dim), dtype=tf.float32)
      else:
        initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              common_layers.shape_list(initial_output))
    target_modality = self._problem_hparams.target_modality
    if target_modality.is_class_modality:
      decode_length = 1
    else:
      if "partial_targets" in features:
        prefix_length = common_layers.shape_list(features["partial_targets"])[1]
      else:
        prefix_length = common_layers.shape_list(features["inputs"])[1]
      decode_length = prefix_length + decode_length

    # Initial values of result, logits and loss.
    result = initial_output
    if self._target_modality_is_real:
      logits = tf.zeros((batch_size, 0, 1, target_modality.top_dimensionality))
      logits_shape_inv = [None, None, None, None]
    else:
      # tensor of shape [batch_size, time, 1, 1, vocab_size]
      logits = tf.zeros((batch_size, 0, 1, 1,
                         target_modality.top_dimensionality))
      logits_shape_inv = [None, None, None, None, None]
    if not tf.contrib.eager.in_eager_mode():
      logits.set_shape(logits_shape_inv)

    loss = 0.0

    result, logits, loss, enc_out = infer_step(result, logits, loss)

    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    # Reassign targets back to the previous value.
    if targets_old is not None:
      features["targets"] = targets_old
    losses = {"training": loss}
    if "partial_targets" in features:
      partial_target_length = common_layers.shape_list(
          features["partial_targets"])[1]
      result = tf.slice(result, [0, partial_target_length, 0, 0],
                        [-1, -1, -1, -1])

    # Modified
    # Added encoder outputs

    return {
        "outputs": result,
        "scores": None,
        "encoder_outputs": enc_out,
        "logits": logits,
        "losses": losses,
    }

  def sample(self, features):
    """Run the model and extract samples.

    Args:
      features: an map of string to `Tensor`.

    Returns:
       samples: an integer `Tensor`.
       logits: a list of `Tensor`s, one per datashard.
       losses: a dictionary: {loss-name (string): floating point `Scalar`}.
    """
    # Modified
    # Passing encoder output reference

    logits, losses, enc_out = self(features)  # pylint: disable=not-callable
    if self._target_modality_is_real:
      return logits, logits, losses  # Raw numbers returned from real modality.
    if self.hparams.sampling_method == "argmax":
      samples = tf.argmax(logits, axis=-1)
    else:
      assert self.hparams.sampling_method == "random"

      def multinomial_squeeze(logits, temperature=1.0):
        logits_shape = common_layers.shape_list(logits)
        reshaped_logits = (
            tf.reshape(logits, [-1, logits_shape[-1]]) / temperature)
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(choices, logits_shape[:-1])
        return choices

      samples = multinomial_squeeze(logits, self.hparams.sampling_temp)

    return samples, logits, losses, enc_out