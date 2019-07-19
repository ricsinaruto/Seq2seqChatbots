from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

from t2t_csaky.config import TRANSFORMER_HPARAMS


# change these in config.py
@registry.register_hparams
def general_transformer_hparams():
  hparams = transformer.transformer_base()
  hparams.add_hparam('roulette', TRANSFORMER_HPARAMS['roulette_wheel'])
  hparams.add_hparam('roulette_beam_size',
                     TRANSFORMER_HPARAMS['roulette_beam_size'])

  hparams.batch_size = TRANSFORMER_HPARAMS['batch_size']
  hparams.layer_prepostprocess_dropout = TRANSFORMER_HPARAMS['layer_dropout']
  hparams.symbol_modality_num_shards = TRANSFORMER_HPARAMS['embed_num_shards']
  hparams.attention_dropout = TRANSFORMER_HPARAMS['attention_dropout']
  hparams.relu_dropout = TRANSFORMER_HPARAMS['relu_dropout']
  hparams.summarize_vars = TRANSFORMER_HPARAMS['summarize_vars']

  return hparams


@registry.register_hparams
def general_roulette_transformer_hparams():
  return general_transformer_hparams()


@registry.register_hparams
def general_extracted_transformer_hparams():
  return general_transformer_hparams()


# Exactly replicates the base transformer model described in the paper.
@registry.register_hparams
def chatbot_cornell_base():
  hparams = transformer.transformer_base()
  hparams.learning_rate_warmup_steps = 16000
  return hparams


# Different batch sizes.
@registry.register_hparams
def chatbot_transformer_batch_32k():
  hparams = chatbot_cornell_base()
  hparams.batch_size = 32768
  return hparams


@registry.register_hparams
def chatbot_transformer_batch_16k():
  hparams = chatbot_cornell_base()
  hparams.batch_size = 16384
  return hparams


@registry.register_hparams
def chatbot_transformer_batch_8k():
  hparams = chatbot_cornell_base()
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def chatbot_transformer_batch_4k():
  hparams = chatbot_cornell_base()
  hparams.batch_size = 4096
  return hparams


@registry.register_hparams
def chatbot_transformer_batch_2k():
  hparams = chatbot_cornell_base()
  hparams.batch_size = 2048
  return hparams


# Different dropout values.
@registry.register_hparams
def base_trf_20_10_drop():
  hparams = chatbot_transformer_batch_2k()
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  return hparams


@registry.register_hparams
def base_trf_40_20_drop():
  hparams = chatbot_transformer_batch_2k()
  hparams.layer_prepostprocess_dropout = 0.4
  hparams.attention_dropout = 0.2
  hparams.relu_dropout = 0.2
  return hparams


@registry.register_hparams
def base_trf_50_30_drop():
  hparams = chatbot_transformer_batch_2k()
  hparams.layer_prepostprocess_dropout = 0.5
  hparams.attention_dropout = 0.3
  hparams.relu_dropout = 0.3
  return hparams


@registry.register_hparams
def base_trf_70_50_drop():
  hparams = chatbot_transformer_batch_2k()
  hparams.layer_prepostprocess_dropout = 0.7
  hparams.attention_dropout = 0.5
  hparams.relu_dropout = 0.5
  return hparams
