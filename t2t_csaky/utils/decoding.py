# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decoding utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import os
import time

import numpy as np
import six

from six.moves import input  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import problem as problem_lib
from tensor2tensor.data_generators import text_encoder
import tensorflow as tf

FLAGS = tf.flags.FLAGS

# Number of samples to draw for an image input (in such cases as captioning)
IMAGE_DECODE_LENGTH = 100


def decode_hparams(overrides=""):
  """Hyperparameters for decoding."""
  hp = tf.contrib.training.HParams(
      save_images=False,
      log_results=True,
      extra_length=100,
      batch_size=0,
      beam_size=4,
      alpha=0.6,
      return_beams=False,
      write_beam_scores=False,
      max_input_size=-1,
      identity_output=False,
      num_samples=-1,
      delimiter="\n",
      decode_to_file=None,
      shards=1,
      shard_id=0,
      force_decode_length=False)
  hp.parse(overrides)
  return hp


def log_encoding_results(inputs,
                         encoder_output,
                         inputs_vocab,
                         identity_output=False):

  """Log encoding results."""

  if identity_output:
    decoded_inputs = " ".join(map(str, inputs.flatten()))
  else:
    decoded_inputs = inputs_vocab.decode(_save_until_eos(inputs, False))

  tf.logging.info("State extraction results INPUT: %s" % decoded_inputs)

  if encoder_output is not None:
    tf.logging.info("State extraction results ENCODER OUTPUT: {}"
                    .format(encoder_output))

  return decoded_inputs


def decode_from_dataset(estimator,
                        problem_name,
                        hparams,
                        decode_hp,
                        decode_to_file=None,
                        dataset_split=None):
  """Perform decoding from dataset."""
  tf.logging.info("Performing local inference from dataset for %s.",
                  str(problem_name))
  # We assume that worker_id corresponds to shard number.
  shard = decode_hp.shard_id if decode_hp.shards > 1 else None

  # Setup the decode output directory for any artifacts that may be written out
  output_dir = os.path.join(estimator.model_dir, "decode")
  tf.gfile.MakeDirs(output_dir)

  # If decode_hp.batch_size is specified, use a fixed batch size
  if decode_hp.batch_size:
    hparams.batch_size = decode_hp.batch_size
    hparams.use_fixed_batch_size = True

  dataset_kwargs = {
      "shard": shard,
      "dataset_split": dataset_split,
      "max_records": decode_hp.num_samples
  }

  # Build the inference input function
  problem = hparams.problem
  infer_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)

  # Get the predictions as an iterable
  predictions = estimator.predict(infer_input_fn)

  # Prepare output file writers if decode_to_file passed
  decode_to_file = decode_to_file or decode_hp.decode_to_file
  if decode_to_file:
    if decode_hp.shards > 1:
      decode_filename = decode_to_file + ("%.2d" % decode_hp.shard_id)
    else:
      decode_filename = decode_to_file

    # TODO fix filename

    output_filepath = _decode_filename(decode_filename, problem_name, decode_hp)
    parts = output_filepath.split(".")
    parts[-1] = "targets"
    target_filepath = ".".join(parts)
    parts[-1] = "inputs"
    input_filepath = ".".join(parts)
    parts[-1] = "enc_state"
    encoder_state_file_path = ".".join(parts)

    input_file = tf.gfile.Open(input_filepath, "w")

  problem_hparams = hparams.problem_hparams
  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  has_input = "inputs" in problem_hparams.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = problem_hparams.vocabulary[inputs_vocab_key]

  encoder_outputs = []
  decoded_inputs = []

  for num_predictions, prediction in enumerate(predictions):
    num_predictions += 1
    inputs = prediction["inputs"]
    encoder_output = prediction["encoder_outputs"]
    decoded_input = log_encoding_results(
        inputs,
        encoder_output,
        inputs_vocab,
        identity_output=decode_hp.identity_output)

    encoder_outputs.append(encoder_output)
    decoded_inputs.append(decoded_input)

    # Write out predictions if decode_to_file passed
    if decode_to_file:
      for i, (e_output, d_input) in \
              enumerate(zip(encoder_outputs, decoded_inputs)):

        input_file.write("{}:\t{}".
                         format(i, str(d_input) + decode_hp.delimiter))

      np.save(encoder_state_file_path, np.array(encoder_outputs))

    if (decode_hp.num_samples >= 0 and
            num_predictions >= decode_hp.num_samples):
      break

  if decode_to_file:
    input_file.close()

  run_postdecode_hooks(DecodeHookArgs(
      estimator=estimator,
      problem=problem,
      output_dir=output_dir,
      hparams=hparams,
      decode_hparams=decode_hp))

  tf.logging.info("Completed inference on %d samples."
                  % num_predictions)  # pylint: disable=undefined-loop-variable


def decode_from_file(estimator,
                     filename,
                     hparams,
                     decode_hp,
                     decode_to_file=None,
                     checkpoint_path=None):
  """Compute predictions on entries in filename and write them out."""
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" %
        decode_hp.batch_size)

  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  problem_name = FLAGS.problem
  tf.logging.info("Performing decoding from a file.")
  sorted_inputs, sorted_keys = _get_sorted_inputs(filename, decode_hp.shards,
                                                  decode_hp.delimiter)
  num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

  def input_fn():
    input_gen = _decode_batch_input_fn(num_decode_batches, sorted_inputs,
                                       inputs_vocab, decode_hp.batch_size,
                                       decode_hp.max_input_size)
    gen_fn = make_input_fn_from_generator(input_gen)
    example = gen_fn()
    return _decode_input_tensor_to_features_dict(example, hparams)

  decoded_inputs = []
  encoder_outputs = []
  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  start_time = time.time()
  total_time_per_step = 0
  total_cnt = 0

  def timer(gen):
    while True:
      try:
        start_time = time.time()
        item = next(gen)
        elapsed_time = time.time() - start_time
        yield elapsed_time, item
      except StopIteration:
        break

  for elapsed_time, result in timer(result_iter):
    decoded_input = log_encoding_results(
        result["inputs"],
        result["encoder_outputs"],
        inputs_vocab)
    decoded_inputs.append(decoded_input)
    encoder_outputs.append(np.array(result["encoder_outputs"]))

    total_time_per_step += elapsed_time
    total_cnt += result["outputs"].shape[-1]
  tf.logging.info("Elapsed Time: %5.5f" % (time.time() - start_time))
  tf.logging.info("Averaged Single Token Generation Time: %5.7f" %
                  (total_time_per_step / total_cnt))

  # Reversing the decoded inputs and outputs because they were reversed in
  # _decode_batch_input_fn
  decoded_inputs.reverse()
  encoder_outputs.reverse()
  # If decode_to_file was provided use it as the output filename without change
  # (except for adding shard_id if using more shards for decoding).
  # Otherwise, use the input filename plus model, hp, problem, beam, alpha.
  decode_filename = decode_to_file if decode_to_file else filename

  if decode_hp.shards > 1:
    decode_filename += "%.2d" % decode_hp.shard_id
  if not decode_to_file:
    decode_filename = _decode_filename(decode_filename,
                                       problem_name,
                                       decode_hp)

  base = os.path.basename(decode_filename).split('.')
  dirname = os.path.dirname(decode_filename)
  encode_filename = os.path.join(dirname, '{}-enc{}'.format(base[0], '.npy'))

  tf.logging.info("Writing inputs into %s" % decode_filename)
  tf.logging.info("Writing encoder outputs into %s" % encode_filename)
  print("Writing encoder outputs into %s" % encode_filename)
  outfile = tf.gfile.Open(decode_filename, "w")

  if decode_to_file:
    for i, (e_output, d_input) in \
            enumerate(zip(encoder_outputs, decoded_inputs)):
      outfile.write("{}".format(str(d_input) + decode_hp.delimiter))

    np.save(encode_filename, np.array(encoder_outputs))

  if decode_to_file:
    outfile.close()


def _decode_filename(base_filename, problem_name, decode_hp):
  return "{base}.{model}.{hp}.{problem}.beam{beam}.alpha{alpha}.decodes"\
      .format(
      base=base_filename,
      model=FLAGS.model,
      hp=FLAGS.hparams_set,
      problem=problem_name,
      beam=str(decode_hp.beam_size),
      alpha=str(decode_hp.alpha))


def make_input_fn_from_generator(gen):
  """Use py_func to yield elements from the given generator."""
  first_ex = six.next(gen)
  flattened = tf.contrib.framework.nest.flatten(first_ex)
  types = [t.dtype for t in flattened]
  shapes = [[None] * len(t.shape) for t in flattened]
  first_ex_list = [first_ex]

  def py_func():
    if first_ex_list:
      example = first_ex_list.pop()
    else:
      example = six.next(gen)
    return tf.contrib.framework.nest.flatten(example)

  def input_fn():
    flat_example = tf.py_func(py_func, [], types)
    _ = [t.set_shape(shape) for t, shape in zip(flat_example, shapes)]
    example = tf.contrib.framework.nest.pack_sequence_as(first_ex,
                                                         flat_example)
    return example

  return input_fn


def decode_interactively(estimator, hparams, decode_hp, checkpoint_path=None):
  """Interactive decoding."""

  def input_fn():
    gen_fn = make_input_fn_from_generator(
        _interactive_input_fn(hparams, decode_hp))
    example = gen_fn()
    example = _interactive_input_tensor_to_features_dict(example, hparams)
    return example

  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)
  for result in result_iter:
    is_image = False  # TODO(lukaszkaiser): find out from problem id / class.
    targets_vocab = hparams.problem_hparams.vocabulary["targets"]

    if decode_hp.return_beams:
      beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      scores = None
      if "scores" in result:
        scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
      for k, beam in enumerate(beams):
        tf.logging.info("BEAM %d:" % k)
        beam_string = targets_vocab.decode(_save_until_eos(beam, is_image))
        if scores is not None:
          tf.logging.info("\"%s\"\tScore:%f" % (beam_string, scores[k]))
        else:
          tf.logging.info("\"%s\"" % beam_string)
    else:
      if decode_hp.identity_output:
        tf.logging.info(" ".join(map(str, result["outputs"].flatten())))
      else:
        tf.logging.info(
            targets_vocab.decode(_save_until_eos(result["outputs"], is_image)))


def _decode_batch_input_fn(num_decode_batches, sorted_inputs, vocabulary,
                           batch_size, max_input_size):
  """Generator to produce batches of inputs."""
  tf.logging.info(" batch %d" % num_decode_batches)
  # First reverse all the input sentences so that if you're going to get OOMs,
  # you'll see it in the first batch
  sorted_inputs.reverse()
  for b in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
      input_ids = vocabulary.encode(inputs)
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      input_ids.append(text_encoder.EOS_ID)
      batch_inputs.append(input_ids)
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length
      x = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(x)

    yield {
        "inputs": np.array(final_batch_inputs).astype(np.int32),
    }


def _interactive_input_fn(hparams, decode_hp):
  """Generator that reads from the terminal and yields "interactive inputs".
  Due to temporary limitations in tf.learn, if we don't want to reload the
  whole graph, then we are stuck encoding all of the input as one fixed-size
  numpy array.
  We yield int32 arrays with shape [const_array_size].  The format is:
  [num_samples, decode_length, len(input ids), <input ids>, <padding>]
  Args:
    hparams: model hparams
    decode_hp: decode hparams
  Yields:
    numpy arrays
  Raises:
    Exception: when `input_type` is invalid.
  """
  num_samples = decode_hp.num_samples if decode_hp.num_samples > 0 else 1
  decode_length = decode_hp.extra_length
  input_type = "text"
  p_hparams = hparams.problem_hparams
  has_input = "inputs" in p_hparams.input_modality
  vocabulary = p_hparams.vocabulary["inputs" if has_input else "targets"]
  # This should be longer than the longest input.
  const_array_size = 10000
  # Import readline if available for command line editing and recall.
  try:
    import readline  # pylint: disable=g-import-not-at-top,unused-variable
  except ImportError:
    pass
  while True:
    prompt = ("INTERACTIVE MODE  num_samples=%d  decode_length=%d  \n"
              "  it=<input_type>     ('text' or 'image' or 'label', default: "
              "text)\n"
              "  ns=<num_samples>    (changes number of samples, default: 1)\n"
              "  dl=<decode_length>  (changes decode length, default: 100)\n"
              "  <%s>                (decode)\n"
              "  q                   (quit)\n"
              ">" % (num_samples, decode_length, "source_string"
                     if has_input else "target_prefix"))
    input_string = input(prompt)
    if input_string == "q":
      return
    elif input_string[:3] == "ns=":
      num_samples = int(input_string[3:])
    elif input_string[:3] == "dl=":
      decode_length = int(input_string[3:])
    elif input_string[:3] == "it=":
      input_type = input_string[3:]
    else:
      if input_type == "text":
        input_ids = vocabulary.encode(input_string)
        if has_input:
          input_ids.append(text_encoder.EOS_ID)
        x = [num_samples, decode_length, len(input_ids)] + input_ids
        assert len(x) < const_array_size
        x += [0] * (const_array_size - len(x))
        features = {
            "inputs": np.array(x).astype(np.int32),
        }
      elif input_type == "image":
        input_path = input_string
        img = vocabulary.encode(input_path)
        features = {
            "inputs": img.astype(np.int32),
        }
      elif input_type == "label":
        input_ids = [int(input_string)]
        x = [num_samples, decode_length, len(input_ids)] + input_ids
        features = {
            "inputs": np.array(x).astype(np.int32),
        }
      else:
        raise Exception("Unsupported input type.")
      for k, v in six.iteritems(
          problem_lib.problem_hparams_to_features(p_hparams)):
        features[k] = np.array(v).astype(np.int32)
      yield features


def save_video(video, save_path_template):
  """Save frames of the videos into files."""
  try:
    from PIL import Image  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    tf.logging.warning(
        "Showing and saving an image requires PIL library to be "
        "installed: %s", e)
    raise NotImplementedError("Image display and save not implemented.")

  for i, frame in enumerate(video):
    save_path = save_path_template.format(i)
    with tf.gfile.Open(save_path, "wb") as sp:
      Image.fromarray(np.uint8(frame)).save(sp)


def show_and_save_image(img, save_path):
  """Shows an image using matplotlib and saves it."""
  try:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    tf.logging.warning(
        "Showing and saving an image requires matplotlib to be "
        "installed: %s", e)
    raise NotImplementedError("Image display and save not implemented.")
  plt.imshow(img)
  with tf.gfile.Open(save_path, "wb") as sp:
    plt.savefig(sp)


def _get_sorted_inputs(filename, num_shards=1, delimiter="\n"):
  """Returning inputs sorted according to length.
  Args:
    filename: path to file with inputs, 1 per line.
    num_shards: number of input shards. If > 1, will read from file
    filename.XX,
      where XX is FLAGS.worker_id.
    delimiter: str, delimits records in the file.
  Returns:
    a sorted list of inputs
  """
  tf.logging.info("Getting sorted inputs")
  # read file and sort inputs according them according to input length.
  if num_shards > 1:
    decode_filename = filename + ("%.2d" % FLAGS.worker_id)
  else:
    decode_filename = filename

  with tf.gfile.Open(decode_filename) as f:
    text = f.read()
    records = text.split(delimiter)
    inputs = [record.strip() for record in records]
    # Strip the last empty line.
    if not inputs[-1]:
      inputs.pop()
  input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


def _save_until_eos(hyp, is_image):
  """Strips everything after the first <EOS> token, which is normally 1."""
  hyp = hyp.flatten()
  if is_image:
    return hyp
  try:
    index = list(hyp).index(text_encoder.EOS_ID)
    return hyp[0:index]
  except ValueError:
    # No EOS_ID: return the array as-is.
    return hyp


def _interactive_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.
  Args:
    feature_map: dict with inputs.
    hparams: model hyperparameters
  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.convert_to_tensor(feature_map["inputs"])
  input_is_image = False if len(inputs.get_shape()) < 3 else True

  x = inputs
  if input_is_image:
    x = tf.image.resize_images(x, [299, 299])
    x = tf.reshape(x, [1, 299, 299, -1])
    x = tf.to_int32(x)
  else:
    # Remove the batch dimension.
    num_samples = x[0]
    length = x[2]
    x = tf.slice(x, [3], tf.to_int32([length]))
    x = tf.reshape(x, [1, -1, 1, 1])
    # Transform into a batch of size num_samples to get that many random
    # decodes.
    x = tf.tile(x, tf.to_int32([num_samples, 1, 1, 1]))

  p_hparams = hparams.problem_hparams
  input_space_id = tf.constant(p_hparams.input_space_id)
  target_space_id = tf.constant(p_hparams.target_space_id)

  features = {}
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (
      IMAGE_DECODE_LENGTH if input_is_image else inputs[1])
  features["inputs"] = x
  return features


def _decode_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.
  Args:
    feature_map: dict with inputs.
    hparams: model hyperparameters
  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.convert_to_tensor(feature_map["inputs"])
  input_is_image = False

  x = inputs
  p_hparams = hparams.problem_hparams
  # Add a third empty dimension
  x = tf.expand_dims(x, axis=[2])
  x = tf.to_int32(x)
  input_space_id = tf.constant(p_hparams.input_space_id)
  target_space_id = tf.constant(p_hparams.target_space_id)

  features = {}
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (
      IMAGE_DECODE_LENGTH if input_is_image else tf.shape(x)[1] + 50)
  features["inputs"] = x
  return features


def latest_checkpoint_step(ckpt_dir):
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if not ckpt:
    return None
  path = ckpt.model_checkpoint_path
  step = int(path.split("-")[-1])
  return step


class DecodeHookArgs(collections.namedtuple(
    "DecodeHookArgs",
    ["estimator", "problem", "output_dir", "hparams", "decode_hparams"])):
  pass


def run_postdecode_hooks(decode_hook_args):
  """Run hooks after decodes have run."""
  hooks = decode_hook_args.problem.decode_hooks
  if not hooks:
    return
  global_step = latest_checkpoint_step(decode_hook_args.estimator.model_dir)
  if global_step is None:
    tf.logging.info(
        "Skipping decode hooks because no checkpoint yet available.")
    return
  tf.logging.info("Running decode hooks.")
  summary_writer = tf.summary.FileWriter(decode_hook_args.output_dir)
  for hook in hooks:
    # Isolate each hook in case it creates TF ops
    with tf.Graph().as_default():
      summaries = hook(decode_hook_args)
    if summaries:
      summary = tf.Summary(value=list(summaries))
      summary_writer.add_summary(summary, global_step)
  summary_writer.close()
  tf.logging.info("Decode hooks done.")

