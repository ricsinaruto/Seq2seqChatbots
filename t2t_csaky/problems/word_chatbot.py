from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators.text_problems import VocabType
from tensor2tensor.utils import metrics
from tensor2tensor.layers import modalities

from t2t_csaky.config import PROBLEM_HPARAMS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


class WordChatbot(text_problems.Text2TextProblem):
  '''
  An abstract base class for word based chatbot problems.
  '''

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def is_generate_per_split(self):
    return True

  @property
  def vocab_file(self):
    return self.vocab_filename

  @property
  def vocab_filename(self):
    return 'vocab.chatbot.' + str(self.targeted_vocab_size)

  @property
  def oov_token(self):
    return '<unk>'

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def targeted_vocab_size(self):
    return PROBLEM_HPARAMS['vocabulary_size']

  @property
  def targeted_dataset_size(self):
    # Number of utterance pairs in the full dataset.
    # If it's 0, then the full size of the dataset is used.
    return PROBLEM_HPARAMS['dataset_size']

  @property
  def dataset_split(self):
    return PROBLEM_HPARAMS['dataset_split']

  @property
  def dataset_splits(self):
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': PROBLEM_HPARAMS['num_train_shards'],
    }, {
        'split': problem.DatasetSplit.EVAL,
        'shards': PROBLEM_HPARAMS['num_dev_shards'],
    }, {
        'split': problem.DatasetSplit.TEST,
        'shards': PROBLEM_HPARAMS['num_dev_shards'],
    }]

  @property
  def data_dir(self):
    return ''

  @property
  def raw_data_dir(self):
    return ''

  @property
  def raw_data(self):
    return ''

  @property
  def zipped_data(self):
    return ''

  @property
  def url(self):
    return ''

  ''' Setter methods for the string properties. '''
  @data_dir.setter
  def data_dir(self, value):
    self._data_dir = value

  @raw_data_dir.setter
  def raw_data_dir(self, value):
    self._raw_data_dir = value

  @raw_data.setter
  def raw_data(self, value):
    self._raw_data = value

  @zipped_data.setter
  def zipped_data(self, value):
    self._zipped_data = value

  @url.setter
  def url(self, value):
    self._url = value

  # Main function where the preprocessing of the data starts.
  def preprocess_data(self, train_mode):
    return NotImplementedError

  # hparams for the problem.
  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.stop_at_eos = int(True)

    p.modality = {'targets': modalities.ModalityType.SYMBOL}
    if self.has_inputs:
      p.modality['inputs'] = modalities.ModalityType.SYMBOL
      p.vocab_size = {'inputs': self._encoders['inputs'].vocab_size}
    p.vocab_size['targets'] = self._encoders['inputs'].vocab_size

    if self.vocab_type == VocabType.CHARACTER:
      p.loss_multiplier = 2.0

    if self.packed_length:
      if self.has_inputs:
        p.modality['inputs_segmentation'] = modalities.ModalityType.IDENTITY
        p.modality['inputs_position'] = modalities.ModalityType.IDENTITY
        p.vocab_size['inputs_segmentation'] = None
        p.vocab_size['inputs_position'] = None
      p.modality['targets_segmentation'] = modalities.ModalityType.IDENTITY
      p.modality['targets_position'] = modalities.ModalityType.IDENTITY
      p.vocab_size['targets_segmentation'] = None
      p.vocab_size['targets_position'] = None

  # What evaluation metrics to use with this problem.
  def eval_metrics(self):
    return [metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
            metrics.Metrics.ACC_PER_SEQ,
            metrics.Metrics.NEG_LOG_PERPLEXITY,
            metrics.Metrics.APPROX_BLEU]

  # Override this, to start with preprocessing.
  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    self.data_dir = data_dir
    # Determine whether we are in training or validation mode.
    self.mode = {problem.DatasetSplit.TRAIN: 'train',
                 problem.DatasetSplit.EVAL: 'dev',
                 problem.DatasetSplit.TEST: 'test'}
    filepath_fns = {problem.DatasetSplit.TRAIN: self.training_filepaths,
                    problem.DatasetSplit.EVAL: self.dev_filepaths,
                    problem.DatasetSplit.TEST: self.test_filepaths}

    split_paths = [(split['split'], filepath_fns[split['split']](
      data_dir, split['shards'], shuffled=self.already_shuffled))
      for split in self.dataset_splits]
    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    if self.is_generate_per_split:
      for split, paths in split_paths:
        # Create the source and target txt files from the raw data.
        self.preprocess_data(self.mode[split])
        generator_utils.generate_files(
            self.generate_encoded_samples(data_dir, tmp_dir, split), paths)
    else:
      self.preprocess_data(self.mode[problem.DatasetSplit.TRAIN])
      generator_utils.generate_files(
          self.generate_encoded_samples(
              data_dir, tmp_dir, problem.DatasetSplit.TRAIN), all_paths)

    generator_utils.shuffle_dataset(all_paths, extra_fn=self._pack_fn())

  # This function generates train and validation pairs in t2t-datagen style.
  def generate_samples(self, data_dir, tmp_dir, data_split):
    '''
    The function assumes that if you have data at one level of the pipeline,
    you don't want to re-generate it, so for example if the 4 txt files exist,
    the function continues by generating the t2t-datagen format files.
    So if you want to re-download or re-generate data,
    you have to delete it first from the appropriate directories.

    Params:
      :data_dir: Directory where the data will be generated
                 The raw data has to be downloaded one directory level higher.
      :data_split: Which data split to generate samples for.
    '''
    self.data_dir = data_dir
    print('t2t_csaky_log: ' +
          self.mode[data_split] + ' data generation activated.')

    sPath = os.path.join(data_dir, self.mode[data_split] + 'Source.txt')
    tPath = os.path.join(data_dir, self.mode[data_split] + 'Target.txt')

    # Open the files and yield source-target lines.
    with tf.gfile.GFile(sPath, mode='r') as source_file:
      with tf.gfile.GFile(tPath, mode='r') as target_file:
        source, target = source_file.readline(), target_file.readline()
        while source and target:
          yield {'inputs': source.strip(), 'targets': target.strip()}
          source, target = source_file.readline(), target_file.readline()

  # Save the vocabulary to a file.
  def save_vocab(self, vocab):
    '''
    Params:
      :vocab: Vocabulary list.
    '''
    voc_file = open(os.path.join(self._data_dir, self.vocab_file), 'w')

    # Put the reserved tokens in.
    voc_file.write('<pad>\n')
    voc_file.write('<EOS>\n')
    for word, _ in vocab.most_common(self.targeted_vocab_size - 3):
      voc_file.write(word + '\n')
    voc_file.write('<unk>')

    voc_file.close()

  # Open the 6 files to write the processed data into.
  def open_6_files(self):
    trainSource = open(os.path.join(self._data_dir, 'trainSource.txt'), 'w')
    trainTarget = open(os.path.join(self._data_dir, 'trainTarget.txt'), 'w')
    devSource = open(os.path.join(self._data_dir, 'devSource.txt'), 'w')
    devTarget = open(os.path.join(self._data_dir, 'devTarget.txt'), 'w')
    testSource = open(os.path.join(self._data_dir, 'testSource.txt'), 'w')
    testTarget = open(os.path.join(self._data_dir, 'testTarget.txt'), 'w')

    return trainSource, trainTarget, devSource, \
        devTarget, testSource, testTarget

  # Close the 6 files to write the processed data into.
  def close_n_files(self, files):
    for file in files:
      file.close()
