""" 
This file is the main python file of the project importing all my
  problem,
  model, 
  hparam registrations
"""

from t2t_csaky.problems import character_chatbot
from t2t_csaky.problems import cornell_chatbots
from t2t_csaky.problems import daily_dialog_chatbot
from t2t_csaky.problems import persona_chat_chatbot

from t2t_csaky.models import roulette_transformer
from t2t_csaky.models import extracted_transformer
from t2t_csaky.models import gradient_checkpointed_seq2seq

from t2t_csaky.hparams import transformer_hparams
from t2t_csaky.hparams import seq2seq_hparams