from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# general imports
import tensorflow as tf
import os
import requests
import tarfile
import re
import gzip
import zipfile
from collections import Counter
from clint.textui import progress

# tensor2tensor imports
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

# my imports
from t2t_csaky.problems import word_chatbot

# Flags
FLAGS = tf.flags.FLAGS

# End-of-sentence marker
EOS = text_encoder.EOS_ID


@registry.register_problem
class OpensubtitlesChatbot(word_chatbot.WordChatbot):
  """
  A class implementing the chatbot problem for the OpenSubtitles dataset.
  """

  @property
  def num_shards(self):
    return 100

  @property
  def num_dev_shards(self):
    return 100

  @property
  def targeted_vocab_size(self):
    return 100000

  @property
  def targeted_dataset_size(self):
    return 0

  @property
  def dataset_version(self):
    # year of the opensubtitles dataset creation
    return 2011

  @property
  def dataset_split(self):
    return {"train":65,"val":25,"test":10}

  # main function where the preprocessing of the data starts
  def preprocess_data(self, train_mode):
    """
    Params:
      :train_mode: whether we are in train or dev mode
    """

    year="" if self.dataset_version==2009 else str(self.dataset_version)
    # set the raw data directory and data
    self.raw_data_dir=os.path.join("/".join(self._data_dir.split("/")[:-1]), \
                                    'raw_data_'+str(self.dataset_version))
    self.raw_data=os.path.join(self._raw_data_dir, "OpenSubtitles"+year)
    self.zipped_data=os.path.join(self._raw_data_dir,"en.tar.gz")

    # create the download url
    self.url="http://opus.nlpl.eu/download.php?f=OpenSubtitles"+str(year)+"/en.tar.gz"

    # check at which part of the pipeline are we at
    self.data_pipeline_status(train_mode)

  # check at which part of the pipeline are we at
  def data_pipeline_status(self, train_mode):
    """
    This function first check recursively at which point in the
    data processing point are we (what files can be found on the disk),
    and then proceeds from there.

    Params:
      :train_mode: whether we are in train or dev mode
    """

    # build the source and target paths
    sourcePath=os.path.join(self._data_dir, train_mode+"Source.txt")
    targetPath=os.path.join(self._data_dir, train_mode+"Target.txt")

    # if raw data dir doesn't exist, create it
    if not os.path.exists(self._raw_data_dir):
      os.makedirs(self._raw_data_dir)

    # check whether sourcePath.txt exists
    if os.path.isfile(sourcePath) and os.path.isfile(targetPath) and \
      os.path.isfile(os.path.join(self._data_dir, self.vocab_file)):
      print("t2t_csaky_log: Source, target and vocab files exist in "+self._data_dir+", proceeding "+\
              "with data generation. If you want to rebuild these files, delete them first.")
      return
    
    # check whether the raw data is extracted to the raw_data_dir folder
    elif os.path.exists(self._raw_data):
      print("t2t_csaky_log: No source, target or vocab files found in "+self._data_dir+".")
      print("t2t_csaky_log: Extracted raw data found in "+self._raw_data_dir+". Proceeding "+\
                            "with creating source, target and vocab files.")
      self.create_data(train_mode)

    # check whether the data is downloaded in the raw_data_dir_folder
    elif os.path.exists(self._zipped_data):
      print("t2t_csaky_log: No source, target or vocab files found in "+self._data_dir+".")
      print("t2t_csaky_log: No extracted raw data found in "+self._raw_data_dir+".")
      print("t2t_csaky_log: Unextracted raw data found in "+self._raw_data_dir+". Proceeding "+\
                            "with extracting and creating source, target and vocab files.")

      self.extract_data(train_mode)

    else:
      print("t2t_csaky_log: No source, target or vocab files found in "+self._data_dir+".")
      print("t2t_csaky_log: No raw data found in "+self._raw_data_dir+". Proceeding "+\
                            "with downloading the data, extracting it, and creating "+\
                            "source, target and vocab files.")
      self.download_data(train_mode)

  # download data from official sources
  def download_data(self, train_mode):
    """
    Params:
      :train_mode:  whether we are in train or dev mode
    """

    # open the url and download the data with progress bars
    data_stream = requests.get(self._url, stream=True)
    with open(self._zipped_data, 'wb') as file:
      total_length = int(data_stream.headers.get('content-length'))
      for chunk in progress.bar(data_stream.iter_content(chunk_size=1024), \
                                expected_size=total_length/1024 + 1): 
        if chunk:
          file.write(chunk)
          file.flush()

    # next step is extracting the data
    print("t2t_csaky_log: Extracting data to "+self._zipped_data+".")
    self.extract_data(train_mode)

  # extract data and go to the next step
  def extract_data(self, train_mode):
    """
    Params:
      :train_mode:  whether we are in train or dev mode
    """

    if self._zipped_data[-2:]=="gz":
      zip_file=tarfile.open(self._zipped_data, "r:gz")
    elif self._zipped_data[-3:]=="zip":
      zip_file = zipfile.ZipFile(self._zipped_data, 'r')
    else:
      print("t2t_csaky_log: "+self._zipped_data+" is not a .zip or .gz file, so I can't extract it.")

    zip_file.extractall(self._raw_data_dir)
    zip_file.close()

    # next step is creating the source, target and vocab files
    print("t2t_csaky_log: Creating "+train_mode+" files in "+self._data_dir+".")
    self.create_data(train_mode)

  # create the source, target and vocab files
  def create_data(self, train_mode):
    """
    Params:
      :train_mode: whether we are in train or dev mode
    """

    # open the 6 files
    trainSource, trainTarget, devSource, devTarget, testSource, testTarget = self.open_6_files()

    conv_id = 0
    number_of_lines=0
    dataset_split_counter=0
    vocabulary=Counter()
    # find all the files
    for root, subfolders, files in os.walk(self._raw_data_dir):
      for file in files:
        if conv_id % 100==0:
          print("t2t_csaky_log: Parsed "+str(conv_id)+" files.")
        if file.endswith('.gz'):
          source_lines=''
          target_lines=''
          conv_id += 1
          dataset_split_counter+=1
          
          # open one .gz file and parse it
          with gzip.open(os.path.join(root, file), 'r') as txt_file:
            words = ''
            line_id = 1

            # parse one line
            for line in txt_file:
              line = str(line)

              # check if it's a new sentence
              if line.find('<s id="') != -1:
                if len(words) > 0:
                  # do some cleaning
                  words=self.clean_line(words)

                  # build the vocabulary
                  if dataset_split_counter<=self.dataset_split["train"]:
                    word_list=words.split()
                    for word in word_list:
                      if word in vocabulary:
                        vocabulary[word]+=1
                      else:
                        vocabulary[word]=1
                  
                  # add the previous line
                  source_lines += words + '\n'
                  if line_id!=1:
                    target_lines += words + '\n'
                  line_id += 1
                words = ''

              else:
                index = line.find('<w id="')
                if index >= 0:
                  line = line[index:]
                  word = line[line.find('>')+1:line.find('</w')]
                  words = words + ' ' + word.replace('\t', ' ')

            # delete the final source sentence, since it doesn't have a target
            source_lines='\n'.join(source_lines.split('\n')[:-2])+'\n'

          # save the dialog according to the dataset split
          if dataset_split_counter<=self.dataset_split["train"]:
            trainSource.write(source_lines)
            trainTarget.write(target_lines)
          elif dataset_split_counter<=self.dataset_split["train"]+self.dataset_split["val"]:
            devSource.write(source_lines)
            devTarget.write(target_lines)
          else:
            testSource.write(source_lines)
            testTarget.write(target_lines)

          # reset the split counter if we reached 100%
          if dataset_split_counter == 100:
            dataset_split_counter=0

          # check if we reached the desired dataset size
          number_of_lines+=line_id
          if self.targeted_dataset_size!=0 and self.targeted_dataset_size<number_of_lines:
            break

    # close the files
    self.close_6_files(trainSource, trainTarget, devSource, devTarget, testSource, testTarget)

    # save the vocabulary
    self.save_vocab(vocabulary)

  # save the vocabulary to a file
  def save_vocab(self, vocab):
    """ 
    Params:
      :vocab: vocabulary list
    """
    voc_file=open(os.path.join(self._data_dir, self.vocab_file), 'w')

    # put the reserved tokens in
    voc_file.write("<pad>\n")
    voc_file.write("<EOS>\n")
    for word, _ in vocab.most_common(self.targeted_vocab_size-3):
      voc_file.write(word+'\n')
    voc_file.write("<UNK>")

    voc_file.close()

  # clean a line with some re rules
  def clean_line(self, line):
    """
    Params:
      :line: line to be processed and returned
    """
    line = line.lower()
    line = re.sub("[^a-z .!?'\t\\\]", "", line)
    line = re.sub("\\\['] ", " '", line)
    line = re.sub("[\\\]", " ", line)
    line = re.sub("[.]", " . ", line)
    line = re.sub("[?]", " ? ", line)
    line = re.sub("[!]", " ! ", line)
    line = re.sub("[ ]'[ ]"," ",line)
    line = re.sub("n't"," n't",line)

    return line

  # open the 6 files to write the processed data into
  def open_6_files(self):
    trainSource = open(os.path.join(self._data_dir, 'trainSource.txt'), 'w')
    trainTarget = open(os.path.join(self._data_dir, 'trainTarget.txt'), 'w')
    devSource = open(os.path.join(self._data_dir, 'devSource.txt'), 'w')
    devTarget = open(os.path.join(self._data_dir, 'devTarget.txt'), 'w')
    testSource = open(os.path.join(self._data_dir, 'testSource.txt'), 'w')
    testTarget = open(os.path.join(self._data_dir, 'testTarget.txt'), 'w')

    return trainSource, trainTarget, devSource, devTarget, testSource, testTarget

  # close the 6 files to write the processed data into
  def close_6_files(self, trainSource, trainTarget, devSource, devTarget, testSource, testTarget):
    trainSource.close()
    trainTarget.close()
    devSource.close()
    devTarget.close()
    testSource.close()
    testTarget.close() 