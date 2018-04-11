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
from t2t_csaky.problems import opensubtitles_chatbot
from t2t_csaky.config import *


# End-of-sentence marker
EOS = text_encoder.EOS_ID


@registry.register_problem
class CornellChatbotBasic(opensubtitles_chatbot.OpensubtitlesChatbot):
  """
  A class implementing the chatbot problem for the Cornell Movie Dialog dataset.
  """

  # main function where the preprocessing of the data starts
  def preprocess_data(self, train_mode):
    """
    Params:
      :train_mode: whether we are in train or dev mode
    """

    # set the raw data directory and data
    self.raw_data_dir=os.path.join("/".join(self._data_dir.split("/")[:-1]),'raw_data')
    self.raw_data=os.path.join(self._raw_data_dir, "cornell movie-dialogs corpus")
    self.zipped_data=os.path.join(self._raw_data_dir,"cornell_movie_dialogs_corpus.zip")

    # create the download url
    self.url="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"

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
    movie_lines = open(os.path.join(self._raw_data, 'movie_lines.txt'), errors="ignore")
    dialog_list = self.extract_dialog_ids()

    vocabulary=Counter()
    line_dict={}
    number_of_lines=0
    # iterate through file
    for line in movie_lines:
      if number_of_lines % 10000==0:
        print("t2t_csaky_log: Parsed "+str(number_of_lines)+" lines.")

      line=line.split(" +++$+++ ")
      dialog_id=line[0]
      line=line[4].lower()

      # do some cleaning
      line=self.clean_line(line)
      line_dict[dialog_id]=line

      number_of_lines+=1
      # check if we reached the desired dataset size
      if self.targeted_dataset_size!=0 and self.targeted_dataset_size<number_of_lines:
        break

    counter=0
    dataset_split_counter=0
    # save the actual dialogs
    for dialog in dialog_list:
      if counter % 10000==0:
        print("t2t_csaky_log: Saved "+str(counter)+"/"+str(len(dialog_list))+" dialogs.")

      dataset_split_counter+=1
      i=0
      # save one utterance
      for utterance in dialog:
        if utterance != dialog[-1] and dialog[i+1]!="L211194" and dialog[i+1]!="L1045":
          source_line=line_dict[utterance]+'\n'
          target_line=line_dict[dialog[i+1]]+'\n'

          # save to the files according to dataset split
          if dataset_split_counter<=self.dataset_split["train"]:
            # build vocabulary
            words=source_line.split()
            for word in words:
              if word in vocabulary:
                vocabulary[word]+=1
              else:
                vocabulary[word]=1

            trainSource.write(source_line)
            trainTarget.write(target_line)

          elif dataset_split_counter<=self.dataset_split["train"]+self.dataset_split["val"]:
            devSource.write(source_line)
            devTarget.write(target_line)
          else:
            testSource.write(source_line)
            testTarget.write(target_line)
        i+=1

      # reset the split counter if we reached 100%
      if dataset_split_counter == 100:
        dataset_split_counter=0
      counter+=1

    # close the files
    self.close_6_files(trainSource, trainTarget, devSource, devTarget, testSource, testTarget)
    movie_lines.close()

    # save the vocabulary
    self.save_vocab(vocabulary)

  # clean a line with some re rules
  def clean_line(self, line):
    """
    Params:
      :line: line to be processed and returned 
    """ 

    # 2 functions for more complex replacing
    def replace(matchobj):
      return re.sub("'"," '",str(matchobj.group(0)))
    def replace_null(matchobj):
      return re.sub("'","",str(matchobj.group(0)))

    # keep some special tokens
    line = re.sub("[^a-z .?!'0-9]", "", line)
    line =re.sub("[.]", " . ", line)
    line =re.sub("[?]", " ? ", line)
    line =re.sub("[!]", " ! ", line)

    # take care of apostrophes
    line=re.sub("[ ]'[ ]", " ", line)
    line=re.sub(" '[a-z]", replace_null, line)
    line=re.sub("n't", " n't", line)
    line=re.sub("[^ n]'[^ t]", replace, line)

    return line

  # extract the dialog ids from the dialog file
  def extract_dialog_ids(self):
    dialogs = open(os.path.join(self._raw_data, 'movie_conversations.txt'), errors="ignore")

    dialog_list=[]
    # each line contains a dialog
    for line in dialogs:
      line=line.split(" +++$+++ ")
      line=line[3].split(",")

      i=0
      for item in line:
        line[i]=re.sub("[^A-Z0-9]","",item)
        i+=1
      dialog_list.append(line)

    dialogs.close()
    return dialog_list


@registry.register_problem
class CornellChatbotSeparateNames(CornellChatbotBasic):
  """
  A class implementing the chatbot problem for the Cornell Movie Dialog dataset with
  the names of the characters saying a line appended to that line.
  """

  @property
  def targeted_name_vocab_size(self):
    return PROBLEM_HPARAMS["name_vocab_size"]

  @property
  def targeted_vocab_size(self):
    return PROBLEM_HPARAMS["vocabulary_size"] + PROBLEM_HPARAMS["name_vocab_size"]

  # create the source, target and vocab files
  def create_data(self, train_mode):
    """
    Params:
      :train_mode: whether we are in train or dev mode
    """

    # open the 6 files
    trainSource, trainTarget, devSource, devTarget, testSource, testTarget = self.open_6_files()

    # open the raw data
    movie_lines = open(os.path.join(self._raw_data, 'movie_lines.txt'), errors="ignore")
    dialog_list = self.extract_dialog_ids()

    vocabulary=Counter()
    name_vocab=Counter()
    line_dict={}
    number_of_lines=0
    # iterate through file
    for line in movie_lines:
      if number_of_lines % 10000==0:
        print("t2t_csaky_log: Parsed "+str(number_of_lines)+" lines.")

      line=line.split(" +++$+++ ")

      # it's important to separate between characters with same names but appearing in different movies
      name=re.sub(" ", "_", line[3])+"_"+line[2]
      dialog_id=line[0]
      line=line[4].lower()

      # build vocabulary for names:
      # Currently we build it based on the whole dataset, because we can assume
      # that the list of most frequent names is the same in the whole dataset, 
      # and in a random sample of it, however it would be more accurate to
      # build the name vocab based solely on the training examples.
      if name in name_vocab:
        name_vocab[name]+=1
      elif name!="":
        name_vocab[name]=1

      # do some cleaning
      line=self.clean_line(line)
      line_dict[dialog_id]=name+" "+line

      number_of_lines+=1
      # check if we reached the desired dataset size
      if self.targeted_dataset_size!=0 and self.targeted_dataset_size<number_of_lines:
        break

    # replace infrequent names with unknown
    line_dict=self.replace_names(line_dict, name_vocab)

    # save the actual dialogs
    counter=0
    dataset_split_counter=0
    for dialog in dialog_list:
      if counter % 10000==0:
        print("t2t_csaky_log: Saved "+str(counter)+"/"+str(len(dialog_list))+" dialogs.")

      dataset_split_counter+=1
      i=0
      # save one utterance
      for utterance in dialog:
        if utterance != dialog[-1] and dialog[i+1]!="L211194" and dialog[i+1]!="L1045":

          # prepare the name annotated data
          target_words=line_dict[dialog[i+1]].split()
          target_name=target_words[0]
          target_line=" ".join(target_words[1:])+'\n'
          source_line=line_dict[utterance]+" "+target_name+'\n'

          # save to the files according to dataset split
          if dataset_split_counter<=self.dataset_split["train"]:
            # build vocabulary
            words=source_line.split()[1:-1]
            for word in words:
              if word in vocabulary:
                vocabulary[word]+=1
              else:
                vocabulary[word]=1

            trainSource.write(source_line)
            trainTarget.write(target_line)

          elif dataset_split_counter<=self.dataset_split["train"]+self.dataset_split["val"]:
            devSource.write(source_line)
            devTarget.write(target_line)
          else:
            testSource.write(source_line)
            testTarget.write(target_line)
        i+=1

      # reset the split counter if we reached 100%
      if dataset_split_counter == 100:
        dataset_split_counter=0
      counter+=1

    # close the files
    self.close_6_files(trainSource, trainTarget, devSource, devTarget, testSource, testTarget)
    movie_lines.close()

    # save the vocabulary
    self.save_vocab(vocabulary, name_vocab)

  # replace infrequent names with unknown
  def replace_names(self, line_dict, name_vocab):
    """
    Params:
      :line_dict:   dictionary containing all the parsed lines
      :name_vocab:  the vocabulary of names
    """

    name_list=[]
    for name, _ in name_vocab.most_common(self.targeted_name_vocab_size-1):
      name_list.append(name)

    for dialog_id in line_dict:
      line=line_dict[dialog_id].split()

      if line[0] not in name_list:
        string=" "+line[0]+" "
        line_dict[dialog_id]=re.sub(string, " <unk_name> ", " "+line_dict[dialog_id]+" ")

    return line_dict

  # save the vocabulary to a file
  def save_vocab(self, vocab, name_vocab):
    """ 
    Params:
      :vocab:       vocabulary list
      :name_vocab:  name vocabulary
    """
    voc_file=open(os.path.join(self._data_dir, self.vocab_file), 'w')

    # put the reserved tokens in
    voc_file.write("<pad>\n")
    voc_file.write("<EOS>\n")

    # basic words
    for word, _ in vocab.most_common(self.targeted_vocab_size-3):
      voc_file.write(word+'\n')
    voc_file.write("<unk>"+'\n')

    # name vocab
    for name, _ in name_vocab.most_common(self.targeted_name_vocab_size-1):
      voc_file.write(name+'\n')
    voc_file.write("<unk_name>")

    voc_file.close()