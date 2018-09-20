import os
import sys
import tensorflow as tf

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.bin import t2t_decoder
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils import decoding

FLAGS = tf.flags.FLAGS

# Other setup.
Modes = tf.estimator.ModeKeys


def decode(estimator, hparams, decode_hp):
  """Decode from estimator. Interactive, from file, or from dataset."""
  if FLAGS.decode_interactive:
    if estimator.config.use_tpu:
      raise ValueError("TPU can only decode from dataset.")
    decoding.decode_interactively(estimator, hparams, decode_hp,
                                  checkpoint_path=FLAGS.checkpoint_path)
  elif FLAGS.decode_from_file:
    if estimator.config.use_tpu:
      raise ValueError("TPU can only decode from dataset.")
    decoding.decode_from_file(estimator, FLAGS.decode_from_file, hparams,
                              decode_hp, FLAGS.decode_to_file,
                              checkpoint_path=FLAGS.checkpoint_path)
    if FLAGS.checkpoint_path and FLAGS.keep_timestamp:
      ckpt_time = os.path.getmtime(FLAGS.checkpoint_path + ".index")
      os.utime(FLAGS.decode_to_file, (ckpt_time, ckpt_time))
  else:
    decoding.decode_from_dataset(
        estimator,
        FLAGS.problem,
        hparams,
        decode_hp,
        decode_to_file=FLAGS.decode_to_file,
        dataset_split="test" if FLAGS.eval_use_test_set else None)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    trainer_lib.set_random_seed(FLAGS.random_seed)
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

    hp = t2t_decoder.create_hparams()
    decode_hp = t2t_decoder.create_decode_hparams()

    estimator = trainer_lib.create_estimator(
        FLAGS.model,
        hp,
        t2t_trainer.create_run_config(hp),
        decode_hparams=decode_hp,
        use_tpu=FLAGS.use_tpu)

    decode(estimator, hp, decode_hp)


if __name__ == '__main__':
    tf.app.run()
