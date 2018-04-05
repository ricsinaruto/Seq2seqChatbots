from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tensor2tensor imports
from tensor2tensor.models import lstm
from tensor2tensor.utils import registry



""" Only this works with own_hparams_seq2seq model, so it has to be changed to the appropriate batch size. """
def chatbot_lstm_hparams():
  hparams=chatbot_lstm_batch_256()
  hparams.hidden_size=2048
  return hparams


""" Different batch sizes. """
@registry.register_hparams
def chatbot_lstm_batch_8k():
  hparams = lstm.lstm_seq2seq()
  hparams.max_length = 256
  hparams.clip_grad_norm = 0.
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.shared_embedding_and_softmax_weights=True
  hparams.optimizer="Adafactor"

  hparams.symbol_modality_num_shards=10
  hparams.hidden_size=2048
  hparams.num_hidden_layers=2
  hparams.batch_size=8192
  return hparams

@registry.register_hparams
def chatbot_lstm_batch_1k():
  hparams = chatbot_lstm_batch_8k()
  hparams.batch_size=1024
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