from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# general imports
import tensorflow as tf
import os
import re
from collections import Counter

# tensor2tensor imports
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

# my imports
from t2t_csaky.problems import cornell_chatbots

# Flags
FLAGS = tf.flags.FLAGS

# End-of-sentence marker
EOS = text_encoder.EOS_ID



@registry.register_problem
class DailyDialogChatbot(cornell_chatbots.CornellChatbotBasic):
  """
  A class implementing a simple turn-based chatbot problem for the DailyDialog dataset.
  This version doesn't use any auxiliary information.
  """
  @property
  def num_shards(self):
    return 1

  @property
  def num_dev_shards(self):
    return 1

  @property
  def targeted_vocab_size(self):
    return 16384

  @property
  def targeted_dataset_size(self):
    return 0

  @property
  def dataset_split(self):
    return {"train":80,"val":10,"test":10}

  # main function where the preprocessing of the data starts
  def preprocess_data(self, train_mode):
    """
    Params:
      :train_mode: whether we are in train or dev mode
    """

    # set the raw data directory and data
    self.raw_data_dir=os.path.join("/".join(self._data_dir.split("/")[:-1]),'raw_data')
    self.raw_data=os.path.join(self._raw_data_dir, "ijcnlp_dailydialog")
    self.zipped_data=os.path.join(self._raw_data_dir,"ijcnlp_dailydialog.zip")

    # create the download url
    self.url="http://yanran.li/files/ijcnlp_dailydialog.zip"

    # check at which part of the pipeline are we at
    self.data_pipeline_status(train_mode)

  # create the source, target and vocab files
  def create_data(self, train_mode):
    """
    Params:
      :train_mode: whether we are in train or dev mode
    """

    # open the 6 files
    trainSource, trainTarget, devSource, devTarget, testSource, testTarget = self.open_6_files()

    # open the raw data
    dialogs=open(os.path.join(self._raw_data, 'dialogues_text.txt'), errors="ignore")

    vocabulary=Counter()
    number_of_dialogs=0
    dataset_split_counter=0
    # iterate through the file
    for dialog in dialogs:
      dataset_split_counter+=1
      if number_of_dialogs % 1000 == 0:
        print("t2t_csaky_log: Parsed "+str(number_of_dialogs)+" dialogs.")

      # utterances are separated by the __eou__ token
      utterances=dialog.split("__eou__")

      # check which file we should write to
      if dataset_split_counter<=self.dataset_split["train"]:
        source_file=trainSource
        target_file=trainTarget
      elif dataset_split_counter<=self.dataset_split["train"]+self.dataset_split["val"]:
        source_file=devSource
        target_file=devTarget
      else:
        source_file=testSource
        target_file=testTarget

      # clean the utterances
      i=0
      for utterance in utterances:
        utterance=self.clean_line(utterance.lower())
        i+=1

        # build vocabulary
        if dataset_split_counter<=self.dataset_split["train"]:
          words=utterance.split()[:-1]
          for word in words:
            if word in vocabulary:
              vocabulary[word]+=1
            else:
              vocabulary[word]=1

        # write to files
        if i!=len(utterances):
          source_file.write(utterance+"\n")
        if i!=1:
          target_file.write(utterance+"\n")

      number_of_dialogs+=1
      # reset the split counter if we reached 100%
      if dataset_split_counter == 100:
        dataset_split_counter=0

    # close the files
    self.close_6_files(trainSource, trainTarget, devSource, devTarget, testSource, testTarget)

    # save the vocabulary
    self.save_vocab(vocabulary)