import argparse
import os
import datetime
import subprocess
import sys
import select

# my imports
from config import *
from data_filtering.hash_jaccard import HashJaccard
from data_filtering.meaning_based_clustering import RNNState
from data_filtering.sentence_embedding import SentenceEmbedding
from data_filtering.identity_clustering import IdentityClustering



# save the config.py file for a specific run
def save_config_file(directory):
  # make the data dir if it doesn't exist
  if not os.path.exists(directory):
    os.makedirs(directory)

  # this will be used in the names of saved files
  now=datetime.datetime.now()
  time_string=(str(now.year)+"."
              +str(now.month)+"."
              +str(now.day)+"."
              +str(now.hour)+"."
              +str(now.minute)+"."
              +str(now.second))

  os.system("cp "+FLAGS["t2t_usr_dir"]+"/config.py "+directory+"/config."
            +time_string+".txt")

# initialize a data generation problem
def data_generating():
  print("Program is running in data generation mode.")
  save_config_file(FLAGS["data_dir"])
  os.system("t2t-datagen \
               --t2t_usr_dir="+FLAGS["t2t_usr_dir"]
            +" --data_dir="+FLAGS["data_dir"]
            +" --problem="+FLAGS["problem"])

# initialize a training loop with the given flags
def training():
  print("Program is running in training mode.")
  save_config_file(FLAGS["train_dir"])
  # what hparams should we use
  if FLAGS["hparams"]=="":
    hparam_string="general_"+FLAGS["model"]+"_hparams"
  else:
    hparam_string=FLAGS["hparams"]

  os.system("t2t-trainer \
               --generate_data=False \
               --t2t_usr_dir="+FLAGS["t2t_usr_dir"]
            +" --data_dir="+FLAGS["data_dir"]
            +" --problem="+FLAGS["problem"]
            +" --output_dir="+FLAGS["train_dir"]
            +" --model="+FLAGS["model"]
            +" --hparams_set="+hparam_string
            +" --schedule="+FLAGS["train_mode"]
            +" --worker_gpu_memory_fraction="+str(FLAGS["memory_fraction"])
            +" --keep_checkpoint_max="+str(FLAGS["keep_checkpoints"])
            +" --keep_checkpoint_every_n_hours="+str(FLAGS["save_every_n_hour"])
            +" --save_checkpoints_secs="+str(FLAGS["save_every_n_secs"])
            +" --train_steps="+str(FLAGS["train_steps"])
            +" --eval_steps="+str(FLAGS["evaluation_steps"])
            +" --local_eval_frequency="+str(FLAGS["evaluation_freq"]))

# intialize an inference test with the given flags
def decoding():
  print("Program is running in inference/decoding mode.")
  save_config_file(FLAGS["decode_dir"])
  # what hparams should we use
  if FLAGS["hparams"]=="":
    hparam_string="general_"+FLAGS["model"]+"_hparams"
  else:
    hparam_string=FLAGS["hparams"]

  decode_mode_string=""
  # determine the decode mode flag
  if FLAGS["decode_mode"]=="interactive":
    decode_mode_string=" --decode_interactive"
  elif FLAGS["decode_mode"]=="file":
    decode_mode_string=(" --decode_from_file="
                        +FLAGS["decode_dir"]+"/"
                        +FLAGS["input_file_name"])

  os.system("t2t-decoder \
               --generate_data=False \
               --t2t_usr_dir="+FLAGS["t2t_usr_dir"]
            +" --data_dir="+FLAGS["data_dir"]
            +" --problem="+FLAGS["problem"]
            +" --output_dir="+FLAGS["train_dir"]
            +" --model="+FLAGS["model"]
            +" --worker_gpu_memory_fraction="+str(FLAGS["memory_fraction"])
            +" --hparams_set="+hparam_string
            +" --decode_to_file="+FLAGS["decode_dir"]+"/"
                                 +FLAGS["output_file_name"]
            +' --decode_hparams="beam_size='+str(FLAGS["beam_size"])
                                +",return_beams="+FLAGS["return_beams"]+'"'
            +decode_mode_string)


# initialize a filtering problem
def data_filtering():
  print("Program is running in data filtering mode.")
  save_config_file(DATA_FILTERING["data_dir"])

  filter_problems= {
    "hash_jaccard"      : HashJaccard,
    "sentence_embedding": SentenceEmbedding,
    "rnn_state"         : RNNState,
    "identity_clustering":IdentityClustering
  }

  problem=filter_problems[DATA_FILTERING["filter_problem"]]("full")
  problem.run()

# run a longer experiment, with many calls to the above functions
def experiment():
  clusters=[100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
  for dataset in ["DailyDialog", "Persona_Chat"]:
    for source_cluster in clusters:
      for target_cluster in clusters:
          # modify config files
          DATA_FILTERING["source_clusters"]=source_cluster
          DATA_FILTERING["target_clusters"]=target_cluster
          DATA_FILTERING["data_dir"] = (
            "data_dir/"+dataset
            +"/base_with_numbers/filtered_data/hash_jaccard/"
            +str(source_cluster)+"-"+str(target_cluster)+"_filtering")
          FLAGS["data_dir"]="data_dir/"+dataset+"/base_with_numbers"

          data_filtering()

# run some command line stuff, and get the output in real-time
def run_command(command=["t2t-datagen",
                         "--t2t_usr_dir="+FLAGS["t2t_usr_dir"],
                         "--data_dir="+FLAGS["data_dir"],
                         "--problem="+FLAGS["problem"]]):
  """
  Param:
    :command: List containing the command-line arguments
  """
  process=subprocess.Popen(command,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
  while True:
    reads=[process.stdout.fileno(), process.stderr.fileno()]
    ret=select.select(reads, [], [])

    for fd in ret[0]:
      if fd == process.stdout.fileno():
        read=process.stdout.read(1)
        sys.stdout.write(read.decode("utf-8"))
        sys.stdout.flush()
      if fd == process.stderr.fileno():
        read=process.stderr.read(1)
        sys.stderr.write(read.decode("utf-8"))
        sys.stdout.flush()

    if not read:
      break