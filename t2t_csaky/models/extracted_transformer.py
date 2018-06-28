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

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """Fast decoding.
    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.
    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError(
        "Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.target_modality

    if self.has_input:
      inputs = features["inputs"]
      if target_modality.is_class_modality:
        decode_length = 1
      else:
        decode_length = (
          common_layers.shape_list(inputs)[1] + features.get(
            "decode_length", decode_length))

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]
      input_modality = self._problem_hparams.input_modality["inputs"]
      with tf.variable_scope(input_modality.name):
        inputs = input_modality.bottom_sharded(inputs, dp)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
          self.encode,
          inputs,
          features["target_space_id"],
          hparams,
          features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets,
                                                           2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
        partial_targets_length + features.get("decode_length",
                                              decode_length))
      batch_size = partial_targets_shape[0]

    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
        decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
        tf.zeros([1, decode_length, hparams.hidden_size]),
        hparams.max_length, "body/targets_positional_embedding", None)
    else:
      positional_encoding = None

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.
      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.
      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.
      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
        tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        targets += positional_encoding[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
        decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(
          self.decode,
          targets,
          cache.get("encoder_output"),
          cache.get("encoder_decoder_attention_bias"),
          bias,
          hparams,
          cache,
          nonpadding=features_to_nonpadding(features, "targets"))

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          return tf.one_hot(
            tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
            -1e9)

        ret = tf.cond(
          tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    ret = fast_decode(
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      symbols_to_logits_fn=symbols_to_logits_fn,
      hparams=hparams,
      decode_length=decode_length,
      vocab_size=target_modality.top_dimensionality,
      beam_size=beam_size,
      top_beams=top_beams,
      alpha=alpha,
      batch_size=batch_size,
      force_decode_length=self._decode_hparams.force_decode_length)
    if partial_targets is not None:
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]

    return ret

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

    # batch prediction API.
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
  """Given encoder output and a symbols to logits function, does fast decoding.
  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.
  Args:
    encoder_output: Output from encoder.
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
    symbols_to_logits_fn: Incremental decoding; function mapping triple
      `(ids, step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for longer translations.
    eos_id: End-of-sequence symbol in beam search.
    batch_size: an integer scalar - must be passed if there is no input
    force_decode_length: bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.
  Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    Raises:
      NotImplementedError: If beam size > 1 with partial targets.
  """
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

  return {
    "outputs": decoded_ids,
    "encoder_outputs": encoder_output,
    "scores": scores
  }


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(tf.to_float(features[key]), 1.0)
  return None