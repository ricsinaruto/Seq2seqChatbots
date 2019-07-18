# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Decoding utilities.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import six
import numpy as np
import tensorflow as tf
from queue import Empty

from tensor2tensor.utils import decoding
from tensor2tensor.data_generators import problem as problem_lib
from tensor2tensor.data_generators import text_encoder

FLAGS = tf.flags.FLAGS


def decode_from_dataset(estimator,
                        problem_name,
                        hparams,
                        decode_hp,
                        decode_to_file=None,
                        dataset_split=None):
  '''Perform decoding from dataset.'''
  tf.logging.info('Performing local inference from dataset for %s.',
                  str(problem_name))

  shard = decode_hp.shard_id if decode_hp.shards > 1 else None

  output_dir = os.path.join(estimator.model_dir, 'decode')
  tf.gfile.MakeDirs(output_dir)

  if decode_hp.batch_size:
    hparams.batch_size = decode_hp.batch_size
    hparams.use_fixed_batch_size = True

  dataset_kwargs = {
      'shard': shard,
      'dataset_split': dataset_split,
      'max_records': decode_hp.num_samples
  }

  problem = hparams.problem
  infer_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)

  predictions = estimator.predict(infer_input_fn)

  decode_to_file = decode_to_file or decode_hp.decode_to_file
  if decode_to_file:
    if decode_hp.shards > 1:
      decode_filename = decode_to_file + ('%.2d' % decode_hp.shard_id)
    else:
      decode_filename = decode_to_file

    output_filepath = decoding._decode_filename(decode_filename,
                                                problem_name,
                                                decode_hp)
    parts = output_filepath.split('.')
    parts[-1] = 'targets'
    parts[-1] = 'inputs'
    input_filepath = '.'.join(parts)
    parts[-1] = 'enc_state'
    encoder_state_file_path = '.'.join(parts)

    input_file = tf.gfile.Open(input_filepath, 'w')

  problem_hparams = hparams.problem_hparams
  has_input = 'inputs' in problem_hparams.vocabulary
  inputs_vocab_key = 'inputs' if has_input else 'targets'
  inputs_vocab = problem_hparams.vocabulary[inputs_vocab_key]

  ''' Modified '''
  # Encoder outputs list created.
  encoder_outputs = []
  decoded_inputs = []

  for num_predictions, prediction in enumerate(predictions):
    num_predictions += 1
    inputs = prediction['inputs']
    encoder_output = prediction['encoder_outputs']
    decoded_input = inputs_vocab.decode(
        decoding._save_until_eos(inputs, False))

    encoder_outputs.append(encoder_output)
    decoded_inputs.append(decoded_input)

    ''' Modified '''
    # Writing encoder_outputs list to file.
    if decode_to_file:
      for i, (e_output, d_input) in \
              enumerate(zip(encoder_outputs, decoded_inputs)):
        input_file.write('{}:\t{}'.
                         format(i, str(d_input) + decode_hp.delimiter))

      np.save(encoder_state_file_path, np.array(encoder_outputs))
    if (0 <= decode_hp.num_samples <= num_predictions):
      break

  if decode_to_file:
    input_file.close()

  decoding.decorun_postdecode_hooks(decoding.DecodeHookArgs(
      estimator=estimator,
      problem=problem,
      output_dir=output_dir,
      hparams=hparams,
      decode_hparams=decode_hp))

  tf.logging.info('Completed inference on %d samples.'
                  % num_predictions)  # pylint: disable=undefined-loop-variable


def decode_from_file(estimator,
                     filename,
                     hparams,
                     decode_hp,
                     decode_to_file=None,
                     checkpoint_path=None):
  '''Compute predictions on entries in filename and write them out.'''
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        'decode_hp.batch_size not specified; default=%d' %
        decode_hp.batch_size)

  p_hp = hparams.problem_hparams
  has_input = 'inputs' in p_hp.vocabulary
  inputs_vocab_key = 'inputs' if has_input else 'targets'
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  problem_name = FLAGS.problem
  tf.logging.info('Performing decoding from a file.')
  sorted_inputs, sorted_keys = decoding._get_sorted_inputs(filename,
                                                           decode_hp.shards,
                                                           decode_hp.delimiter)
  num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

  def input_fn():
    input_gen = decoding._decode_batch_input_fn(num_decode_batches,
                                                sorted_inputs,
                                                inputs_vocab,
                                                decode_hp.batch_size,
                                                decode_hp.max_input_size)
    gen_fn = decoding.make_input_fn_from_generator(input_gen)
    example = gen_fn()
    return decoding._decode_input_tensor_to_features_dict(example, hparams)

  ''' Modified '''
  # Encoder outputs list created.
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
    decoded_input = inputs_vocab.decode(
        decoding._save_until_eos(result['inputs'], False))
    decoded_inputs.append(decoded_input)
    encoder_outputs.append(np.array(result['encoder_outputs']))

    total_time_per_step += elapsed_time
    total_cnt += result['outputs'].shape[-1]
  tf.logging.info('Elapsed Time: %5.5f' % (time.time() - start_time))
  tf.logging.info('Averaged Single Token Generation Time: %5.7f' %
                  (total_time_per_step / total_cnt))

  decoded_inputs.reverse()
  encoder_outputs.reverse()

  decode_filename = decode_to_file if decode_to_file else filename

  if decode_hp.shards > 1:
    decode_filename += '%.2d' % decode_hp.shard_id
  if not decode_to_file:
    decode_filename = decoding._decode_filename(decode_filename,
                                                problem_name,
                                                decode_hp)

  base = os.path.basename(decode_filename).split('.')
  dirname = os.path.dirname(decode_filename)
  encode_filename = os.path.join(dirname, '{}{}'.format(base[0], '.npy'))

  tf.logging.info('Writing inputs into %s' % decode_filename)
  tf.logging.info('Writing encoder outputs into %s' % encode_filename)
  print('Writing encoder outputs into %s' % encode_filename)
  outfile = tf.gfile.Open(decode_filename, 'w')

  ''' Modified '''
  # Writing encoder_outputs list to file.
  if decode_to_file:
    for i, (e_output, d_input) in \
            enumerate(zip(encoder_outputs, decoded_inputs)):
      outfile.write('{}'.format(' '.join(
          [word for word in str(d_input).strip().split() if
           word.strip() != '' and word.strip() != '<unk>']) +
          decode_hp.delimiter))

    np.save(encode_filename, np.array(encoder_outputs))

  if decode_to_file:
    outfile.close()


def make_input_fn_from_generator(gen):
  '''Use py_func to yield elements from the given generator.'''
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
    return tf.contrib.framework.nest.pack_sequence_as(first_ex, flat_example)

  return input_fn


def decode_interactively(estimator, hparams, decode_hp,
                         message, response, checkpoint_path=None):
  '''Interactive decoding.'''

  def input_fn():
    gen_fn = make_input_fn_from_generator(
        _interactive_input_fn(hparams, decode_hp, message))
    example = gen_fn()
    return decoding._interactive_input_tensor_to_features_dict(example,
                                                               hparams)

  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)
  for result in result_iter:
    is_image = False  # TODO(lukaszkaiser): find out from problem id / class.
    targets_vocab = hparams.problem_hparams.vocabulary['targets']

    if decode_hp.return_beams:
      beams = np.split(result['outputs'], decode_hp.beam_size, axis=0)
      beam_string = targets_vocab.decode(
          decoding._save_until_eos(beams[0], is_image))
      response.put(beam_string, block=False)
    else:
      if decode_hp.identity_output:
        response.put(' '.join(map(str, result['outputs'].flatten())),
                     block=False)
      else:
        response.put(targets_vocab.decode(decoding._save_until_eos(
            result['outputs'], is_image)), block=False)


def _interactive_input_fn(hparams, decode_hp, message):
  '''Generator that reads from the terminal and yields 'interactive inputs'.

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
  '''
  num_samples = decode_hp.num_samples if decode_hp.num_samples > 0 else 1
  decode_length = decode_hp.extra_length
  input_type = 'text'
  p_hparams = hparams.problem_hparams
  has_input = 'inputs' in p_hparams.input_modality
  vocabulary = p_hparams.vocabulary['inputs' if has_input else 'targets']
  # This should be longer than the longest input.
  const_array_size = 10000
  # Import readline if available for command line editing and recall.
  try:
    import readline  # pylint: disable=g-import-not-at-top,unused-variable
  except ImportError:
    pass
  while True:
    while True:
      try:
        input_string = message.get(block=False)
        break
      except Empty:
        time.sleep(1)

    if input_string == 'q':
      return
    elif input_string[:3] == 'ns=':
      num_samples = int(input_string[3:])
    elif input_string[:3] == 'dl=':
      decode_length = int(input_string[3:])
    elif input_string[:3] == 'it=':
      input_type = input_string[3:]
    else:
      if input_type == 'text':
        input_ids = vocabulary.encode(input_string)
        if has_input:
          input_ids.append(text_encoder.EOS_ID)
        x = [num_samples, decode_length, len(input_ids)] + input_ids
        assert len(x) < const_array_size
        x += [0] * (const_array_size - len(x))
        features = {
            'inputs': np.array(x).astype(np.int32),
        }
      elif input_type == 'image':
        input_path = input_string
        img = vocabulary.encode(input_path)
        features = {
            'inputs': img.astype(np.int32),
        }
      elif input_type == 'label':
        input_ids = [int(input_string)]
        x = [num_samples, decode_length, len(input_ids)] + input_ids
        features = {
            'inputs': np.array(x).astype(np.int32),
        }
      else:
        raise Exception('Unsupported input type.')
      for k, v in six.iteritems(
              problem_lib.problem_hparams_to_features(p_hparams)):
        features[k] = np.array(v).astype(np.int32)
      yield features
