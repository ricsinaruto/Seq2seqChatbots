import argparse
import os
import datetime
import subprocess
import sys
import select
"""
This is the main file, from which any run can be started
"""
# my imports
from config import *

# this will be used in the names of saved files
now=datetime.datetime.now()
my_time_string=str(now.year)+"."+str(now.month)+"."+str(now.day)+"."+str(now.hour)+\
                "."+str(now.minute)+"."+str(now.second)

# save the config.py file for a specific run
def save_config_file(directory):
  os.system("cp "+FLAGS["t2t_usr_dir"]+"/config.py "+directory+"/config."+\
            my_time_string+".txt")

# run some command line stuff, and get the output in real-time
def run_command(command=["t2t-datagen","--t2t_usr_dir="+FLAGS["t2t_usr_dir"],\
                "--data_dir="+FLAGS["data_dir"],"--problem="+FLAGS["problem"]]):
  """
  Param:
    :command: List containing the command-line arguments
  """
  process=subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

# initialize a training loop with the given flags
def run_training():
  # what hparams should we use
  if FLAGS["hparams"]=="":
    hparam_string="general_"+FLAGS["model"]+"_hparams"
  else:
    hparam_string=FLAGS["hparams"]

  os.system("t2t-trainer \
              --generate_data=False \
              --t2t_usr_dir="+FLAGS["t2t_usr_dir"]+\
            " --data_dir="+FLAGS["data_dir"]+\
            " --problems="+FLAGS["problem"]+\
            " --output_dir="+FLAGS["train_dir"]+\
            " --model="+FLAGS["model"]+\
            " --hparams_set="+hparam_string+\
            " --schedule="+FLAGS["train_mode"]+\
            " --keep_checkpoint_max="+str(FLAGS["keep_checkpoints"])+\
            " --keep_checkpoint_every_n_hours="+str(FLAGS["save_every_n_hour"])+\
            " --save_checkpoints_secs="+str(FLAGS["save_every_n_secs"])+\
            " --train_steps="+str(FLAGS["train_steps"])+\
            " --eval_steps="+str(FLAGS["evaluation_steps"])+\
            " --local_eval_frequency="+str(FLAGS["evaluation_freq"]))

# intialize an inference test with the given flags
def run_decoding():
  
  decode_mode_string=""
  # determine the decode mode flag
  if FLAGS["decode_mode"]=="interactive":
    decode_mode_string=" --decode_interactive"
  elif FLAGS["decode_mode"]=="file":
    decode_mode_string=" --decode_from_file="+FLAGS["decode_dir"]+"/"+FLAGS["input_file_name"]

  os.system("t2t-decoder \
              --generate_data=False \
              --t2t_usr_dir="+FLAGS["t2t_usr_dir"]+\
            " --data_dir="+FLAGS["data_dir"]+\
            " --problems="+FLAGS["problem"]+\
            " --output_dir="+FLAGS["train_dir"]+\
            " --model="+FLAGS["model"]+\
            " --hparams_set="+hparam_string+\
            " --decode_to_file="+FLAGS["decode_dir"]+"/"+FLAGS["output_file_name"]+\
            ' --decode_hparams="beam_size='+str(FLAGS["beam_size"])+",return_beams="+FLAGS["return_beams"]+'"'+\
            decode_mode_string)

def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, help='Can be one of the following: {\
                                                train,\
                                                decode,\
                                                generate_data}')
  args=parser.parse_args()
  
  # run in one of 3 modes
  if args.mode=="train":
    print("Program is running in training mode.")
    save_config_file(FLAGS["train_dir"])
    run_training()

  elif args.mode=="decode":
    print("Program is running in inference/decoding mode.")
    save_config_file(FLAGS["decode_dir"])
    run_decoding()

  elif args.mode=="generate_data":
    save_config_file(FLAGS["data_dir"])
    print("Program is running in data generation mode.")
    os.system("t2t-datagen \
                --t2t_usr_dir="+FLAGS["t2t_usr_dir"]+\
              " --data_dir="+FLAGS["data_dir"]+\
              " --problem="+FLAGS["problem"])

  else:
    print("Program exited, because no suitable mode was defined. "+
            "The mode flag has to be set to one of the following:")
    print("  train")
    print("  decode")
    print("  generate_data")

if __name__=="__main__":
  main()