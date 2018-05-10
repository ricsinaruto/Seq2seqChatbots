from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tensor2tensor imports
from tensor2tensor.models import lstm
from tensor2tensor.utils import registry

# my imports
from t2t_csaky.config import *



# change these in config.py
@registry.register_hparams
def general_gradient_checkpointed_seq2seq_hparams():
  hparams=lstm.lstm_seq2seq()

  hparams.clip_grad_norm=0.0
  hparams.shared_embedding_and_softmax_weights = \
    SEQ2SEQ_HPARAMS["shared_embedding_and_softmax_weights"]
  hparams.optimizer=SEQ2SEQ_HPARAMS["optimizer"]
  hparams.use_fixed_batch_size=SEQ2SEQ_HPARAMS["fixed_batch_size"]
  hparams.summarize_vars=SEQ2SEQ_HPARAMS["summarize_vars"]

  hparams.symbol_modality_num_shards=SEQ2SEQ_HPARAMS["embed_num_shards"]
  hparams.hidden_size=SEQ2SEQ_HPARAMS["embedding_size"]
  hparams.num_hidden_layers=SEQ2SEQ_HPARAMS["num_layers"]
  hparams.batch_size=SEQ2SEQ_HPARAMS["batch_size"]
  hparams.max_length = SEQ2SEQ_HPARAMS["max_sentence_len"]
  return hparams

""" From this only the hidden_size is used for the lstm_seq2seq model. """
def chatbot_lstm_hparams():
  hparams=chatbot_lstm_batch_256()
  hparams.hidden_size=SEQ2SEQ_HPARAMS["lstm_hidden_size"]
  return hparams


""" Different batch sizes. """
@registry.register_hparams
def chatbot_lstm_batch_8k():
  hparams = lstm.lstm_seq2seq()

  hparams.clip_grad_norm=0.0
  hparams.shared_embedding_and_softmax_weights=True
  hparams.optimizer="Adafactor"
  hparams.use_fixed_batch_size=False
  hparams.summarize_vars=True

  hparams.symbol_modality_num_shards=10
  hparams.hidden_size=2048
  hparams.num_hidden_layers=2
  hparams.batch_size=8192
  hparams.max_length = 64
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_1():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=1
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_2048():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=2048
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_1024():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=1024
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_4():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=4
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_8():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=8
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_512():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=512
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_256():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=256
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_128():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=128
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_64():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=64
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_32():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=32
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_40():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=40
  return hparams