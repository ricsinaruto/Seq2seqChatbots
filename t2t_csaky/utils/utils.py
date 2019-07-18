import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from config import FLAGS, PROBLEM_HPARAMS


# Temporary helper function to load a vocabulary.
def load_vocab():
  vocab = open(os.path.join(FLAGS['data_dir'],
               'vocab.chatbot.' + str(PROBLEM_HPARAMS['vocabulary_size'])))
  vocab_dict = {}
  # Read the vocab file.
  i = 0
  for word in vocab:
    vocab_dict[word.strip('\n')] = i
    i += 1

  vocab.close()
  return vocab_dict
