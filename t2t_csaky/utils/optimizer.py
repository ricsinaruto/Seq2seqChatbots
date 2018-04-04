from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tensor2tensor imports
from tensor2tensor.utils import yellowfin
from tensor2tensor.utils import optimize as t2t_opt

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer

# gradient checkpointing
from t2t_csaky.utils import memory_saving_gradients



def optimize(loss, learning_rate, hparams):
  """Minimize loss."""
  loss = t2t_opt.weight_decay_and_noise(loss, hparams, learning_rate)
  loss = tf.identity(loss, name="total_loss")
  t2t_opt.log_variable_sizes(verbose=hparams.summarize_vars)
  diet_vars = [
      v for v in tf.global_variables() if v.dtype == dtypes.float16_ref
  ]
  t2t_opt.log_variable_sizes(
      diet_vars, "Diet Variables", verbose=hparams.summarize_vars)
  opt = GradientCheckpointedOptimizer(hparams.optimizer, learning_rate, hparams)

  tf.summary.scalar("learning_rate", learning_rate)
  opt_summaries = ["loss", "global_gradient_norm"]
  if hparams.summarize_grads:
    tf.logging.info("Summarizing gradients")
    opt_summaries.extend(["gradients", "gradient_norm"])

  if hparams.clip_grad_norm:
    tf.logging.info("Clipping gradients, norm: %0.5f", hparams.clip_grad_norm)
  if hparams.grad_noise_scale:
    tf.logging.info("Adding noise to gradients, noise scale: %0.5f",
                    hparams.grad_noise_scale)

  train_op = tf.contrib.layers.optimize_loss(
      name="training",
      loss=loss,
      global_step=tf.train.get_or_create_global_step(),
      learning_rate=learning_rate,
      clip_gradients=hparams.clip_grad_norm or None,
      gradient_noise_scale=hparams.grad_noise_scale or None,
      optimizer=opt,
      summaries=opt_summaries,
      colocate_gradients_with_ops=True)
  return train_op


class GradientCheckpointedOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams, use_tpu=False):
    if optimizer_name == "Adam" and use_tpu:
      # LazyAdamOptimizer does not work on TPU
      optimizer_name = "TrueAdam"

    tf.logging.info("Using optimizer %s", optimizer_name)

    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr,
          momentum=hparams.optimizer_momentum_momentum,
          use_nesterov=hparams.optimizer_momentum_nesterov)
    elif optimizer_name == "YellowFin":
      self._opt = yellowfin.YellowFinOptimizer(
          learning_rate=lr, momentum=hparams.optimizer_momentum_momentum)
    elif optimizer_name == "TrueAdam":
      self._opt = tf.train.AdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Adafactor":
      self._opt = t2t_opt.AdafactorOptimizer(lr / 500.0)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of `loss` for the variables in `var_list`.
    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.
    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.
    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
    """
    if gate_gradients not in [tf.train.Optimizer.GATE_NONE, tf.train.Optimizer.GATE_OP,
                              tf.train.Optimizer.GATE_GRAPH]:
      raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, "
                       "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" %
                       gate_gradients)
    self._assert_valid_dtypes([loss])
    if grad_loss is not None:
      self._assert_valid_dtypes([grad_loss])
    if var_list is None:
      var_list = (
          variables.trainable_variables() +
          ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    else:
      var_list = nest.flatten(var_list)
    # pylint: disable=protected-access
    var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
    # pylint: enable=protected-access
    processors = [optimizer._get_processor(v) for v in var_list]
    if not var_list:
      raise ValueError("No variables to optimize.")
    var_refs = [p.target() for p in processors]
    # TDOD: make the type of gradient checkpointing choosable, but just test it like this first.
    grads = tf.gradients(
        loss, var_refs, grad_ys=grad_loss,
        gate_gradients=(gate_gradients == tf.train.Optimizer.GATE_OP),
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops)
    if gate_gradients == tf.train.Optimizer.GATE_GRAPH:
      grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes(
        [v for g, v in grads_and_vars
         if g is not None and v.dtype != dtypes.resource])
    return grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._opt.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)
