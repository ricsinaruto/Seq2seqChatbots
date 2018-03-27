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

# tensorflow imports
from tensorflow.python.eager import context
from tensorflow.python.util import nest
from tensorflow.python.layers import base

# Flags
FLAGS = tf.flags.FLAGS


@registry.register_model
class RouletteTransformer(transformer.Transformer):
  """
  A child class of the Transformer class, implementing roulette wheel selection.
  """
  
  REGISTERED_NAME="transformer"

  def __init__(self,
      hparams,
      mode,
      problem_hparams=None,
      problem_idx=0,
      data_parallelism=None,
      decode_hparams=None):
    default_name = registry.default_name(type(self))
    name = "transformer"
    base.Layer.__init__(self,trainable=mode == tf.estimator.ModeKeys.TRAIN, name=name)
    if data_parallelism is None:
      data_parallelism = eu.Parallelism([""])
    if problem_hparams is None:
      problem_hparams = hparams.problems[0]

    # If vocabularies differ, unset shared_embedding_and_softmax_weights.
    hparams = copy.copy(hparams)
    if hparams.shared_embedding_and_softmax_weights:
      same_vocab_sizes = True
      for problem in hparams.problems:
        if "inputs" in problem.input_modality:
          if problem.input_modality["inputs"] != problem.target_modality:
            same_vocab_sizes = False
      if not same_vocab_sizes:
        tf.logging.info("Unsetting shared_embedding_and_softmax_weights.")
        hparams.shared_embedding_and_softmax_weights = 0
    self._original_hparams = hparams
    self.set_mode(mode)
    self._decode_hparams = copy.copy(decode_hparams)
    self._data_parallelism = data_parallelism
    self._num_datashards = data_parallelism.n
    self._ps_devices = data_parallelism.ps_devices
    self._problem_hparams = problem_hparams
    self._problem_idx = problem_idx
    self._create_modalities(problem_hparams, self._hparams)
    self._var_store = t2t_model.create_eager_var_store()
    self.attention_weights = dict()  # For vizualizing attention heads.

  def _fast_decode(self,
           features,
           decode_length,
           beam_size=1,
           top_beams=1,
           alpha=1.0):
    """
    Fast decoding.
    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.
    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for slonger translations.
    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams

    inputs = features["inputs"]
    batch_size = common_layers.shape_list(inputs)[0]
    target_modality = self._problem_hparams.target_modality
    if target_modality.is_class_modality:
      decode_length = 1
    else:
      decode_length = common_layers.shape_list(inputs)[1] + decode_length

    # TODO(llion): Clean up this reshaping logic.
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
      inputs = tf.expand_dims(inputs, axis=4)
    s = common_layers.shape_list(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match
    inputs = self._shard_features({"inputs": inputs})["inputs"]
    input_modality = self._problem_hparams.input_modality["inputs"]
    with tf.variable_scope(input_modality.name):
      inputs = input_modality.bottom_sharded(inputs, dp)
    with tf.variable_scope("body"):
      encoder_output, encoder_decoder_attention_bias = dp(
        self.encode, inputs, features["target_space_id"], hparams,
        features=features)
    encoder_output = encoder_output[0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(
        decode_length + 1, hparams.hidden_size)

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

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
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
          self.decode, targets, cache["encoder_output"],
          cache["encoder_decoder_attention_bias"], bias, hparams, cache,
          nonpadding=transformer._features_to_nonpadding(features, "targets"))

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      return tf.squeeze(logits, axis=[1, 2, 3]), cache

    key_channels = hparams.attention_key_channels or hparams.hidden_size
    value_channels = hparams.attention_value_channels or hparams.hidden_size
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

    cache = {
      "layer_%d" % layer: {
        "k": tf.zeros([batch_size, 0, key_channels]),
        "v": tf.zeros([batch_size, 0, value_channels]),
      }
      for layer in range(num_layers)
    }

    # Set 2nd dim to None since it's not invariant in the tf.while_loop
    # Note: Tensor.set_shape() does not work here since it merges shape info.
    # TODO(llion); Find a more robust solution.
    # pylint: disable=protected-access
    if not context.in_eager_mode():
      for layer in cache:
        cache[layer]["k"]._shape = tf.TensorShape([None, None, key_channels])
        cache[layer]["v"]._shape = tf.TensorShape([None, None, value_channels])
    # pylint: enable=protected-access
    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    if beam_size > 1:  # Beam Search
      target_modality = (
        self._hparams.problems[self._problem_idx].target_modality)
      vocab_size = target_modality.top_dimensionality
      initial_ids = tf.zeros([batch_size], dtype=tf.int32)
      decoded_ids, scores = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        stop_early=(top_beams == 1))

      decoded_ids = decoded_ids[:, :, 1:]

      """ t2t_csaky code """
      # do roulette wheel selection or inverse roulette wheel selection
      if self._hparams.roulette=="Normal" or self._hparams.roulette=="Inverse":
        if self._hparams.roulette=="Normal":
          probabilities=tf.pow(tf.constant(2.0),scores)
          start=0
        else:
          probabilities=tf.subtract(tf.constant(1.0),tf.pow(tf.constant(2.0),scores))
          start=beam_size-self._hparams.roulette_beam_size

        ex_probs=tf.divide(probabilities,tf.reduce_sum(probabilities))
        #ex_probs=tf.nn.softmax(probabilities)

        # sample a number between 0 and 1
        wheel=tf.random_uniform([1])
        upper_bound=tf.constant(0.0)

        # change this as well if using inverse
        for i in range(start ,self._hparams.roulette_beam_size):
          upper_bound=tf.add(ex_probs[:,i], upper_bound)
          truthValue=tf.squeeze(tf.logical_and(wheel>=upper_bound-ex_probs[:,i], wheel<=upper_bound))
          decoded_ids,scores,i=tf.cond(
            truthValue,
            lambda: (decoded_ids[:,i,:], scores[:,i], beam_size),
            lambda: (decoded_ids, scores, i)
            )

    else:  # Greedy

      def inner_loop(i, next_id, decoded_ids, cache):
        logits, cache = symbols_to_logits_fn(next_id, i, cache)
        temperature = (0.0 if hparams.sampling_method == "argmax" else
                 hparams.sampling_temp)
        next_id = tf.expand_dims(
          common_layers.sample_with_temperature(logits, temperature), axis=1)
        decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
        return i + 1, next_id, decoded_ids, cache

      decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
      scores = None
      next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
      _, _, decoded_ids, _ = tf.while_loop(
        # TODO(llion): Early stopping.
        lambda i, *_: tf.less(i, decode_length),
        inner_loop,
        [tf.constant(0), next_id, decoded_ids, cache],
        shape_invariants=[
          tf.TensorShape([]),
          tf.TensorShape([None, None]),
          tf.TensorShape([None, None]),
          nest.map_structure(lambda t: tf.TensorShape(t.shape), cache),
        ])

    return decoded_ids, scores