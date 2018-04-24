from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# general imports
import tensorflow as tf
import os
import re
import tarfile
import gzip
import zipfile
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
class PersonaChatChatbot(cornell_chatbots.CornellChatbotBasic):
  """
  A class implementing a simple chatbot for the Persona-chat dataset.
  The personas are not used in this class, only the raw dialogs.
  """

  # main function where the preprocessing of the data starts
  def preprocess_data(self, train_mode):
    """
    Params:
      :train_mode: whether we are in train or dev mode
    """

    # set the raw data directory and data
    self.raw_data_dir=os.path.join("/".join(self._data_dir.split("/")[:-1]),
                                   'raw_data')
    self.raw_data=os.path.join(self._raw_data_dir, "ConvAI2")
    self.zipped_data=os.path.join(self._raw_data_dir,"convai2.tar.gz")

    # create the download url
    self.url="https://s3.amazonaws.com/fair-data/parlai/convai2/convai2.tar.gz"

    # check at which part of the pipeline are we at
    self.data_pipeline_status(train_mode)

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
      print("t2t_csaky_log: "+self._zipped_data
            +" is not a .zip or .gz file, so I can't extract it.")

    zip_file.extractall(self._raw_data)
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
    trainSource, trainTarget, devSource, devTarget, testSource, testTarget = \
      self.open_6_files()

    # open the raw data
    train_dialogs=open(
      os.path.join(self._raw_data, 'train_none_original_no_cands.txt'),
      errors="ignore")
    valid_dialogs=open(
      os.path.join(self._raw_data, 'valid_none_original_no_cands.txt'),
      errors="ignore")
    filenames=[train_dialogs, valid_dialogs]

    # copy the data to a new file
    with open(os.path.join(self._raw_data,
                           'full_none_original_no_cands.txt'), 'w') as outfile:
      for fname in filenames:
        with fname as infile:
          outfile.write(infile.read())
    train_dialogs.close()
    valid_dialogs.close()

    # open the big file
    dialogs=open(
      os.path.join(self._raw_data, 'full_none_original_no_cands.txt'),
      errors="ignore")

    number_of_lines=0
    current_dialog=""
    dialog_list=[]
    dialog_silenced=False
    # iterate through the file and build a list of dialogs separated by __eou__
    for line in dialogs:
      if number_of_lines % 10000 == 0:
        print("t2t_csaky_log: Parsed "+str(number_of_lines)+" lines.")

      dialog_id=line.split()[0]
      # check if this is a refurbished line
      if "__SILENCE__" not in line \
          and ((dialog_silenced and dialog_id=="1") or not dialog_silenced):
        dialog_silenced=False
        number_of_lines+=1
        # get the utterances
        source=" ".join(line.split("\t")[0].split()[1:])
        target=line.split("\t")[1].strip("\n")
        source=self.clean_line(source.lower())
        target=self.clean_line(target.lower())

        # whether this is a new dialog
        if dialog_id=="1" and current_dialog!="":
          dialog_list.append(current_dialog)
          current_dialog=source+"__eou__"+target+"__eou__"
        else:
          current_dialog+=source+"__eou__"+target+"__eou__"
      else:
        dialog_silenced=True

      if self.targeted_dataset_size!=0 and \
          self.targeted_dataset_size<number_of_lines:
        break
    dialogs.close()

    vocabulary=Counter()
    number_of_dialogs=0
    dataset_split_counter=0
    # build the dataset
    for dialog in dialog_list:
      if number_of_dialogs % 1000 == 0:
        print("t2t_csaky_log: Parsed "+str(number_of_dialogs)+" dialogs.")
      
      # check which file we should write to
      if dataset_split_counter<=self.dataset_split["train"]:
        source_file=trainSource
        target_file=trainTarget
      elif dataset_split_counter<=(self.dataset_split["train"]
                                   +self.dataset_split["val"]):
        source_file=devSource
        target_file=devTarget
      else:
        source_file=testSource
        target_file=testTarget

      utterances=dialog.split("__eou__")[:-1]
      i=0
      # loop through the dialog
      for utterance in utterances:
        i+=1
        # build vocabulary
        if dataset_split_counter<=self.dataset_split["train"]:
          words=utterance.split()
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


      dataset_split_counter+=1
      number_of_dialogs+=1
      # reset the split counter if we reached 100%
      if dataset_split_counter == 100:
        dataset_split_counter=0

    # close the files
    self.close_n_files([trainSource,
                        trainTarget,
                        devSource,
                        devTarget,
                        testSource,
                        testTarget])
    # save the vocabulary
    self.save_vocab(vocabulary)