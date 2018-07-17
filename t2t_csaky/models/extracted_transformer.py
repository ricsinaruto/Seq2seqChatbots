from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# general imports
import copy
import re
import tensorflow as tf

# tensor2tensor imports
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import registry
from tensor2tensor.utils import expert_utils as eu

# tensorflow imports
from tensorflow.python.eager import context
from tensorflow.python.util import nest
from tensorflow.python.layers import base


@registry.register_model
class ExtractedTransformer(transformer.Transformer):
  """
  A child class of the Transformer, implementing roulette wheel selection.
  """

  def __init__(self, *args, **kwargs):
    super(ExtractedTransformer, self).__init__(*args, **kwargs)
    self._name = "transformer"

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

    ##### Modified #####
    # Added encoder outputs to predicion dictionary

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

    if "batch_prediction_key" in predictions:
      export_out["batch_prediction_key"] = \
          predictions["batch_prediction_key"]

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


def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                force_decode_length=False):
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  vars_3d_num_heads = (
      hparams.num_heads if hparams.get("attention_variables_3d") else 0)

  cache = {
    "layer_%d" % layer: {
      "k":
        common_attention.split_heads(
            tf.zeros([batch_size, 0, key_channels]), hparams.num_heads),
      "v":
        common_attention.split_heads(
                  tf.zeros([batch_size, 0, value_channels]), hparams.num_heads),
      "f":
        tf.zeros([batch_size, 0, hparams.hidden_size]),
    } for layer in range(num_layers)
  }

  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope(
          "body/decoder/%s/encdec_attention/multihead_attention" % layer_name):
        k_encdec = common_attention.compute_attention_component(
            encoder_output, key_channels, name="k",
            vars_3d_num_heads=vars_3d_num_heads)
        k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
        v_encdec = common_attention.compute_attention_component(
            encoder_output, value_channels, name="v",
            vars_3d_num_heads=vars_3d_num_heads)
        v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

  if beam_size > 1:  # Beam Search
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)
    decoded_ids, scores = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1))

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
      scores = scores[:, 0]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else:  # Greedy

    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """One step of greedy decoding."""
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = (0.0 if hparams.sampling_method == "argmax" else
                     hparams.sampling_temp)
      next_id = common_layers.sample_with_temperature(logits, temperature)
      hit_eos |= tf.equal(next_id, eos_id)

      log_prob_indices = tf.stack(
          [tf.range(tf.to_int64(batch_size)), next_id], axis=1)
      log_prob += tf.gather_nd(log_probs, log_prob_indices)

      next_id = tf.expand_dims(next_id, axis=1)
      decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      finished = i >= decode_length
      if not force_decode_length:
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
    hit_eos = tf.fill([batch_size], False)
    next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    _, _, _, decoded_ids, _, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
            tf.TensorShape([None]),
        ])
    scores = log_prob

  ##### Modified #####
  # Added encoder outputs to predicion dictionary

  return {
    "outputs": decoded_ids,
    "encoder_outputs": encoder_output,
    "scores": scores
  }
