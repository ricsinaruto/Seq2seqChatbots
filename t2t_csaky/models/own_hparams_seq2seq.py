from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tensor2tensor imports
from tensor2tensor.models import lstm
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry

# tensorflow imports
import tensorflow as tf

# my imports
from t2t_csaky.hparams import seq2seq_hparams

# Flags
FLAGS = tf.flags.FLAGS


@registry.register_model
class OwnHparamsSeq2seq(t2t_model.T2TModel):
  """
  A class where I replaced the internal hparams with my own function call.
  This way the hidden_size param of chatbot_lstm_hparams refers to the hidden size
  	of the lstm cells, while the hidden_size specified by the hparam set that is
  	given during training refers to the word embedding size.
  """
  def body(self,features):
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train=self._hparams.mode==tf.estimator.ModeKeys.TRAIN
    return lstm.lstm_seq2seq_internal(
      features.get("inputs"),features["targets"],seq2seq_hparams.chatbot_lstm_hparams(),train)