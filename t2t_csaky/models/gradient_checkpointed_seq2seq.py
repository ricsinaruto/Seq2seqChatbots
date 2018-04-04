from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tensor2tensor imports
from tensor2tensor.models import lstm
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import optimize

# tensorflow imports
import tensorflow as tf
import math

# my imports
from t2t_csaky.hparams import seq2seq_hparams
from t2t_csaky.utils import optimizer

# Flags
FLAGS = tf.flags.FLAGS


@registry.register_model
class GradientCheckpointedSeq2seq(t2t_model.T2TModel):
  """
  A class where I replaced the internal hparams with my own function call.
  This way the hidden_size param of chatbot_lstm_hparams refers to the hidden size
    of the lstm cells, while the hidden_size specified by the hparam set that is
    given during training refers to the word embedding size.

  Moreover, in this class gradient checkpointed is implemented.
  https://github.com/openai/gradient-checkpointing
  """
  def body(self,features):
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train=self._hparams.mode==tf.estimator.ModeKeys.TRAIN
    return lstm.lstm_seq2seq_internal(
      features.get("inputs"),features["targets"],seq2seq_hparams.chatbot_lstm_hparams(),train)

  # Change the optimizer to a new one, which uses gradient checkpointing
  def optimize(self, loss, num_async_replicas=1):
    """Return a training op minimizing loss."""
    tf.logging.info("Base learning rate: %f", self.hparams.learning_rate)
    lr = self.hparams.learning_rate
    decay_rate = optimize.learning_rate_schedule(self.hparams)
    lr *= decay_rate
    if self.hparams.learning_rate_minimum:
      lr_min = float(self.hparams.learning_rate_minimum)
      tf.logging.info("Applying learning rate minimum: %f", lr_min)
      lr = tf.max(lr, tf.to_float(lr_min))
    if num_async_replicas > 1:
      tf.logging.info("Dividing learning rate by num_async_replicas: %d",
                      num_async_replicas)
    lr /= math.sqrt(float(num_async_replicas))
    train_op = optimizer.optimize(loss, lr, self.hparams)
    return train_op
