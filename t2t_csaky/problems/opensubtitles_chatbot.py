from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import requests
import tarfile
import re
import zipfile
import unicodedata
import nltk
from collections import Counter
from clint.textui import progress

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

from t2t_csaky.problems import word_chatbot
from t2t_csaky.config import PROBLEM_HPARAMS


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class OpensubtitlesChatbot(word_chatbot.WordChatbot):
  '''
  A class implementing the chatbot problem for the OpenSubtitles dataset.
  '''

  @property
  def dataset_version(self):
    # Year of the opensubtitles dataset creation.
    return PROBLEM_HPARAMS['dataset_version']

  # Main function where the preprocessing of the data starts.
  def preprocess_data(self, train_mode):
    '''
    Params:
      :train_mode: Whether we are in train or dev mode.
    '''

    year = '' if self.dataset_version == 2009 else str(self.dataset_version)
    # Set the raw data directory and data.
    self.raw_data_dir = os.path.join('/'.join(self._data_dir.split('/')[:-1]),
                                     'raw_data_' + str(self.dataset_version))
    self.raw_data = os.path.join(self._raw_data_dir, 'OpenSubtitles' + year)
    self.zipped_data = os.path.join(self._raw_data_dir, 'en.tar.gz')

    # Create the download url.
    self.url = ('http://opus.nlpl.eu/download.php?f=OpenSubtitles' +
                str(year) + '/en.tar.gz')

    # Check at which part of the pipeline are we at.
    self.data_pipeline_status(train_mode)

  # Check at which part of the pipeline are we at.
  def data_pipeline_status(self, train_mode):
    '''
    This function first check recursively at which point in the
    data processing point are we (what files can be found on the disk),
    and then proceeds from there.

    Params:
      :train_mode: Whether we are in train or dev mode.
    '''

    # Build the source and target paths.
    sourcePath = os.path.join(self._data_dir, train_mode + 'Source.txt')
    targetPath = os.path.join(self._data_dir, train_mode + 'Target.txt')

    # If raw data dir doesn't exist, create it.
    if not os.path.exists(self._raw_data_dir):
      os.makedirs(self._raw_data_dir)

    # Check whether sourcePath.txt exists.
    if (os.path.isfile(sourcePath) and os.path.isfile(targetPath) and
            os.path.isfile(os.path.join(self._data_dir, self.vocab_file))):
      print('t2t_csaky_log: Source, target and vocab files exist in ' +
            self._data_dir + ', proceeding with data generation. ' +
            'If you want to rebuild these files, delete them first.')
      return

    # Check whether the raw data is extracted to the raw_data_dir folder.
    elif os.path.exists(self._raw_data):
      print('t2t_csaky_log: No source, target or vocab files found in ' +
            self._data_dir + '.')
      print('t2t_csaky_log: Extracted raw data is in ' + self._raw_data_dir +
            '. Proceeding with creating source, target and vocab files.')
      self.create_data(train_mode)

    # Check whether the data is downloaded in the raw_data_dir_folder.
    elif os.path.exists(self._zipped_data):
      print('t2t_csaky_log: No source, target or vocab files found in ' +
            self._data_dir + '.')
      print('t2t_csaky_log: No extracted raw data found in ' +
            self._raw_data_dir + '.')
      print('t2t_csaky_log: Unextracted raw data is in ' + self._raw_data_dir +
            '. Extracting and creating source, target and vocab files.')
      self.extract_data(train_mode)

    else:
      print('t2t_csaky_log: No source, target or vocab files found in ' +
            self._data_dir + '.')
      print('t2t_csaky_log: No raw data found in ' + self._raw_data_dir +
            '. Proceeding with downloading the data, extracting it, ' +
            'and creating source, target and vocab files.')
      self.download_data(train_mode)

  # Download data from official sources.
  def download_data(self, train_mode):
    '''
    Params:
      :train_mode:  Whether we are in train or dev mode.
    '''

    # Open the url and download the data with progress bars.
    data_stream = requests.get(self._url, stream=True)
    with open(self._zipped_data, 'wb') as file:
      total_length = int(data_stream.headers.get('content-length'))
      for chunk in progress.bar(data_stream.iter_content(chunk_size=1024),
                                expected_size=total_length / 1024 + 1):
        if chunk:
          file.write(chunk)
          file.flush()

    # Next step is extracting the data.
    print('t2t_csaky_log: Extracting data to ' + self._zipped_data + '.')
    self.extract_data(train_mode)

  # Extract data and go to the next step.
  def extract_data(self, train_mode):
    '''
    Params:
      :train_mode:  Whether we are in train or dev mode.
    '''

    if self._zipped_data[-2:] == 'gz':
      zip_file = tarfile.open(self._zipped_data, 'r:gz')
    elif self._zipped_data[-3:] == 'zip':
      zip_file = zipfile.ZipFile(self._zipped_data, 'r')
    else:
      print('t2t_csaky_log: ' + self._zipped_data +
            ' is not a .zip or .gz file, so I can\'t extract it.')

    zip_file.extractall(self._raw_data_dir)
    zip_file.close()

    # Next step is creating the source, target and vocab files.
    print('t2t_csaky_log: Creating ' +
          train_mode + ' files in ' + self._data_dir)
    self.create_data(train_mode)

  # Create the source, target and vocab files.
  def create_data(self, train_mode):
    '''
    Params:
      :train_mode: Whether we are in train or dev mode.
    '''

    # open the 6 files
    trainSource, trainTarget, devSource, devTarget, testSource, testTarget = \
        self.open_6_files()

    conv_id = 0
    number_of_lines = 0
    dataset_split_counter = 0
    vocabulary = Counter()
    # Dind all the files.
    for root, subfolders, files in os.walk(self._raw_data_dir):
      for file in files:
        if conv_id % 100 == 0:
          print('t2t_csaky_log: Parsed ' + str(conv_id) + ' files.')

        source_lines = ''
        target_lines = ''
        conv_id += 1
        dataset_split_counter += 1

        # Open one .gz file and parse it.
        with open(os.path.join(root, file), 'r', errors='ignore') as txt_file:
          words = ''
          line_id = 1

          # Parse one line.
          for line in txt_file:
            line = str(line)

            # Check if it's a new sentence.
            if line.find('<s id="') != -1:
              # Do some cleaning.
              words = self.clean_line(words)
              if len(words) > 0:
                # Build the vocabulary.
                if dataset_split_counter <= self.dataset_split['train']:
                  word_list = words.split()
                  for word in word_list:
                    if word in vocabulary:
                      vocabulary[word] += 1
                    else:
                      vocabulary[word] = 1

                # Add the previous line.
                source_lines += words + '\n'
                if line_id != 1:
                  target_lines += words + '\n'
                line_id += 1
              words = ''

            else:
              index = line.find('<w id="')
              if index >= 0:
                line = line[index:]
                word = line[line.find('>') + 1:line.find('</w')]
                words = words + ' ' + word.replace('\t', ' ')

          # Delete the final source sentence, since it doesn't have a target.
          source_lines = '\n'.join(source_lines.split('\n')[:-2]) + '\n'

        # Save the dialog according to the dataset split.
        if dataset_split_counter <= self.dataset_split['train']:
          trainSource.write(source_lines)
          trainTarget.write(target_lines)
        elif dataset_split_counter <= (self.dataset_split['train'] +
                                       self.dataset_split['val']):
          devSource.write(source_lines)
          devTarget.write(target_lines)
        else:
          testSource.write(source_lines)
          testTarget.write(target_lines)

        # Reset the split counter if we reached 100%.
        if dataset_split_counter == 100:
          dataset_split_counter = 0

        # Check if we reached the desired dataset size.
        number_of_lines += line_id
        if (self.targeted_dataset_size != 0 and
                self.targeted_dataset_size < number_of_lines):
          break
      else:
        continue
      break

    # Close the files.
    self.close_n_files([trainSource,
                        trainTarget,
                        devSource,
                        devTarget,
                        testSource,
                        testTarget])
    # Save the vocabulary.
    self.save_vocab(vocabulary)

  # Clean a line with some re rules.
  def clean_line(self, line):
    '''
    Params:
      :line: Line to be processed and returned.
    '''
    line = line.lower()
    line = re.sub(' \' ', '\'', line)
    line = unicodedata.normalize('NFKD', line)

    # Keep some special tokens.
    line = re.sub('[^a-z .?!\'0-9]', '', line)
    line = re.sub('n \'t', 'n\'t', line)
    line = re.sub('[.]', ' . ', line)
    line = re.sub('[?]', ' ? ', line)
    line = re.sub('[!]', ' ! ', line)

    words = nltk.word_tokenize(line)
    line = ' '.join(words)
    return line + ' '
