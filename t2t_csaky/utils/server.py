
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from decoding import decode_interactively
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

from flask import Flask, request
from flask_restplus import Resource, Api
import threading
from queue import Queue, Empty
from threading import Lock
import requests
import time
import re

from config import FLAGS

import tensorflow as tf

flags = tf.flags

# Additional flags in bin/t2t_trainer.py and utils/flags.py
tf.flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
tf.flags.DEFINE_string("decode_from_file", None,
                    "Path to the source file for decoding")
tf.flags.DEFINE_string("decode_to_file", None,
                    "Path to the decoded (output) file")
tf.flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
tf.flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
tf.flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
tf.flags.DEFINE_string("score_file", "", "File to score. Each "
                                         "line in the file "
                    "must be in the format input \t target.")
tf.flags.DEFINE_bool("decode_in_memory", False, "Decode in memory.")


def create_hparams():
  return trainer_lib.create_hparams(
      tf.flags.FLAGS.hparams_set,
      tf.flags.FLAGS.hparams,
      data_dir=os.path.expanduser(tf.flags.FLAGS.data_dir),
      problem_name=tf.flags.FLAGS.problem)


def create_decode_hparams():
  hp = tf.contrib.training.HParams(
    save_images=False,
    log_results=True,
    extra_length=100,
    batch_size=0,
    beam_size=10,
    alpha=0.6,
    return_beams=True,
    write_beam_scores=False,
    max_input_size=-1,
    identity_output=False,
    num_samples=-1,
    delimiter="\n",
    decode_to_file=None,
    shards=1,
    shard_id=0,
    force_decode_length=False)
  decode_hp = hp.parse(tf.flags.FLAGS.decode_hparams)
  decode_hp.shards = tf.flags.FLAGS.decode_shards
  decode_hp.shard_id = tf.flags.FLAGS.worker_id
  decode_hp.decode_in_memory = tf.flags.FLAGS.decode_in_memory
  return decode_hp


def clean_line(line):
  """
  Params:
    :line: line to be processed and returned
  """

  # 2 functions for more complex replacing
  def replace(matchobj):
    return re.sub("'"," '",str(matchobj.group(0)))
  def replace_null(matchobj):
    return re.sub("'","",str(matchobj.group(0)))

  # keep some special tokens
  line = re.sub("[^a-z .?!'0-9]", "", line)
  line =re.sub("[.]", " . ", line)
  line =re.sub("[?]", " ? ", line)
  line =re.sub("[!]", " ! ", line)
  line = re.sub("[,]", " , ", line)

  # take care of apostrophes
  line=re.sub("[ ]'[ ]", " ", line)
  line=re.sub(" '[a-z]", replace_null, line)
  line=re.sub("n't", " n't", line)
  line=re.sub("[^ n]'[^ t]", replace, line)

  return line.lower()


def decode(estimator, hparams, decode_hp, message, response):
  """Decode from estimator. Interactive, from file, or from dataset."""
  decode_interactively(estimator, hparams, decode_hp, message, response,
                       tf.flags.FLAGS.checkpoint_path)


def interactive_session(message, response):
  if FLAGS["hparams"] == "":
      hparam_string = "general_" + FLAGS["model"] + "_hparams"
  else:
      hparam_string = FLAGS["hparams"]

  tf.flags.FLAGS.t2t_usr_dir = FLAGS["t2t_usr_dir"]
  tf.flags.FLAGS.data_dir = FLAGS["data_dir"]
  tf.flags.FLAGS.problem = FLAGS["problem"]
  tf.flags.FLAGS.output_dir = FLAGS["train_dir"]
  tf.flags.FLAGS.model = FLAGS["model"]
  tf.flags.FLAGS.worker_gpu_memory_fraction = \
      FLAGS["memory_fraction"]
  tf.flags.FLAGS.hparams_set = hparam_string
  tf.flags.FLAGS.decode_to_file = FLAGS["decode_dir"]+"/" + \
                                   FLAGS["output_file_name"]
  # TODO beam size
  # tf.flags.FLAGS.decode_hparams = '"beam_size='\
  #                                 +str(FLAGS["beam_size"])+\
  #                                 ",return_beams="+FLAGS["return_beams"]+'"'
  tf.flags.FLAGS.decode_interactive = True

  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(tf.flags.FLAGS.random_seed)
  usr_dir.import_usr_dir(tf.flags.FLAGS.t2t_usr_dir)

  hp = create_hparams()
  decode_hp = create_decode_hparams()

  estimator = trainer_lib.create_estimator(
      tf.flags.FLAGS.model,
      hp,
      t2t_trainer.create_run_config(hp),
      decode_hparams=decode_hp,
      use_tpu=tf.flags.FLAGS.use_tpu)

  decode(estimator, hp, decode_hp, message, response)


app = Flask(__name__)
api = Api(app)

PAGE_ACCESS_TOKEN = 'EAAEZAi5NT0ZAwBAD' \
                    'w7sZA7yJBMWQNV0bK' \
                    'kBUhITinuagailF3l' \
                    'FyxpKlsgA41puT2pz' \
                    'ZB3BbyuEzUndMapAN' \
                    '4fD59oqoKFrYMQhoR' \
                    'yLNsc8wK5LpGsiuoz' \
                    'LTIjRDjwSZAtJZB1N' \
                    'ZAmwFLoy9lp5GXFEZ' \
                    'BbxW0Bh3TnEwE5BbZ' \
                    'CteAc3GAiQWWuZBBl'

ns = api.namespace('chatterbee', description='Namespace of the '
                                             'chatter bee app.')


MESSAGE = Queue()
RESPONSE = Queue()


def handle_message(sender_psid, received_message):
  """Handles messages events"""
  if received_message.get('text'):
    MESSAGE.put(clean_line(received_message.get('text')), block=False)
    while True:
      try:
        response = RESPONSE.get(block=True)
        break
      except Empty:
        time.sleep(1)
    response = {
        'text': response
    }
    call_send_api(sender_psid, response)


def handle_postback(sender_psid, received_postback):
  """Handles messaging_postbacks events."""
  pass


def call_send_api(sender_psid, response):
  """Sends response messages via the Send API."""
  request_body = {
      'recipient': {
          'id': sender_psid
      },
      'message': response
  }
  requests.post("https://graph.facebook.com/v2.6/me/messages",
                params={"access_token": PAGE_ACCESS_TOKEN},
                json=request_body)


# noinspection PyMethodMayBeStatic
@ns.route('/')
class Messenger(Resource):

  VERIFY_TOKEN = 'talkingbee'

  def get(self):
    """Verification of messenger web hook."""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    # Checks if a token and mode is in the query string of the request
    if mode is not None and token is not None:

      # Checks the mode and token sent is correct
      if mode == 'subscribe' and token == Messenger.VERIFY_TOKEN:
          # Responds with the challenge token from the request
          return int(challenge)

    api.abort(403)

  def post(self):
    """Handles messaging."""
    body = request.get_json()

    # Checks this is an event from a page subscription
    if body.get('object') == 'page':

      # Iterates over each entry - there may be multiple if batched
      for entry in body['entry']:
        # Gets the message. entry.messaging is an array, but
        # will only ever contain one message, so we get index 0
        messaging = entry.get('messaging')
        if messaging:
          webhook_event = messaging[0]
          sender_id = webhook_event.get('sender').get('id')

          if webhook_event.get('message'):
            handle_message(
                sender_id, webhook_event.get('message'))

          elif webhook_event.get('postback'):
            handle_postback(
                sender_id, webhook_event.get('postback'))


# noinspection PyMethodMayBeStatic
@ns.route('/webhook')
class Webhook(Resource):

  def post(self):
      """Handles webhooks from facebook."""
      print(request.get_json())


if __name__ == '__main__':
  app_thread = threading.Thread(target=app.run)
  service_thread = threading.Thread(
    target=lambda: interactive_session(MESSAGE, RESPONSE))
  service_thread.start()
  app_thread.start()
