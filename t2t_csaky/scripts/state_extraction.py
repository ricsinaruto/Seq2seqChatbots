
import os

from tensor2tensor.bin import t2t_trainer

from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_encoder

from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import t2t_model


import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
flags.DEFINE_string("decode_from_file", None,
                    "Path to the source file for decoding")
flags.DEFINE_string("decode_to_file", None,
                    "Path to the decoded (output) file")
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("score_file", "", "File to score. Each line in the file "
"must be in the format input \t target.")

# Other setup
Modes = tf.estimator.ModeKeys


#
# os.system("t2t-decoder \
#                --generate_data=False \
#                --t2t_usr_dir="+FLAGS["t2t_usr_dir"]
#             +" --data_dir="+FLAGS["data_dir"]
#             +" --problems="+FLAGS["problem"]
#             +" --output_dir="+FLAGS["train_dir"]
#             +" --model="+FLAGS["model"]
#             +" --worker_gpu_memory_fraction="+str(FLAGS["memory_fraction"])
#             +" --hparams_set="+hparam_string
#             +" --decode_to_file="+FLAGS["decode_dir"]+"/"
#                                  +FLAGS["output_file_name"]
#             +' --decode_hparams="beam_size='+str(FLAGS["beam_size"])
#                                 +",return_beams="+FLAGS["return_beams"]+'"'
#             +decode_mode_string)


def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem)


def create_decode_hparams():
  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp.shards = FLAGS.decode_shards
  decode_hp.shard_id = FLAGS.worker_id
  return decode_hp


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


def score_file(filename):
  """Score each line in a file and return the scores."""
  # Prepare model.
  hparams = create_hparams()
  encoders = registry.problem(FLAGS.problem).feature_encoders(FLAGS.data_dir)
  has_inputs = "inputs" in encoders

  # Prepare features for feeding into the model.
  if has_inputs:
    inputs_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
    batch_inputs = tf.reshape(inputs_ph, [1, -1, 1, 1])  # Make it 4D.
  targets_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
  batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])  # Make it 4D.
  features = {
      "inputs": batch_inputs,
      "targets": batch_targets,
  } if has_inputs else {"targets": batch_targets}

  # Prepare the model and the graph when model runs on features.
  model = registry.model(FLAGS.model)(hparams, tf.estimator.ModeKeys.EVAL)
  _, losses = model(features)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Load weights from checkpoint.
    ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
    ckpt = ckpts.model_checkpoint_path
    saver.restore(sess, ckpt)
    # Run on each line.
    results = []
    for line in open(filename):
      tab_split = line.split("\t")
      if len(tab_split) > 2:
        raise ValueError("Each line must have at most one tab separator.")
      if len(tab_split) == 1:
        targets = tab_split[0].strip()
      else:
        targets = tab_split[1].strip()
        inputs = tab_split[0].strip()
      # Run encoders and append EOS symbol.
      targets_numpy = encoders["targets"].encode(
          targets) + [text_encoder.EOS_ID]
      if has_inputs:
        inputs_numpy = encoders["inputs"].encode(inputs) + [text_encoder.EOS_ID]
      # Prepare the feed.
      feed = {
          inputs_ph: inputs_numpy,
          targets_ph: targets_numpy
      } if has_inputs else {targets_ph: targets_numpy}
      # Get the score.
      np_loss = sess.run(losses["training"], feed)
      results.append(np_loss)
  return results


def create_estimator(model_name,
                     hparams,
                     run_config,
                     schedule="train_and_evaluate",
                     decode_hparams=None,
                     use_tpu=False):
  """Create a T2T Estimator."""
  model_fn = t2t_model.T2TModel.make_estimator_model_fn(
      model_name, hparams, decode_hparams=decode_hparams, use_tpu=use_tpu)

  if use_tpu:
    problem = hparams.problem
    batch_size = (
        problem.tpu_batch_size_per_shard(hparams) *
        run_config.tpu_config.num_shards)
    predict_batch_size = batch_size
    if decode_hparams and decode_hparams.batch_size:
      predict_batch_size = decode_hparams.batch_size
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size if "eval" in schedule else None,
        predict_batch_size=predict_batch_size)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=run_config.model_dir, config=run_config)
  return estimator


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    trainer_lib.set_random_seed(FLAGS.random_seed)
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

    if FLAGS.score_file:
        filename = os.path.expanduser(FLAGS.score_file)
        if not tf.gfile.Exists(filename):
            raise ValueError("The file to score doesn't exist: %s" % filename)
        results = score_file(filename)
        if not FLAGS.decode_to_file:
            raise ValueError("To score a file, specify --decode_to_file for results.")
        write_file = open(os.path.expanduser(FLAGS.decode_to_file), "w")
        for score in results:
            write_file.write("%.6f\n" % score)
        write_file.close()
        return

    # 1st

    hp = create_hparams()
    decode_hp = create_decode_hparams()

    estimator = trainer_lib.create_estimator(
        FLAGS.model,
        hp,
        t2t_trainer.create_run_config(hp),
        decode_hparams=decode_hp,
        use_tpu=FLAGS.use_tpu)

    decode(estimator, hp, decode_hp)

    # inputs = None  # INPUTS
    #
    # model = registry.model(model_name)(hparams, Modes.EVAL)
    # model_output = model.infer(inputs)["outputs"]
    #
    # tf.logging.set_verbosity(tf.logging.INFO)
    #
    # # 2nd
    # decode_hp = create_decode_hparams(decode_hparams)
    #
    # estimator = trainer_lib.create_estimator(
    #     model_name,
    #     hparams,
    #     t2t_trainer.create_run_config(hparams),
    #     decode_hparams=decode_hp,
    #     use_tpu=False)
    #
    # decode(estimator, hparams, decode_hp)


if __name__ == '__main__':
    tf.app.run()
