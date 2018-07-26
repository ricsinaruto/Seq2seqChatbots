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
from utils import run


def main():
  # create an argument parser
  parser=argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, help='Can be one of the following: {\
                                                train,\
                                                decode,\
                                                generate_data,\
                                                filter_data,\
                                                experiment}')
  args=parser.parse_args()
  
  # different modes will call different functions
  run_mode={
  "train"             :run.training,
  "decode"            :run.decoding,
  "generate_data"     :run.data_generating,
  "filter_data"       :run.data_filtering,
  "experiment"	      :run.experiment,
  }

  # initialize a mode
  if args.mode in run_mode:
    run_mode[args.mode]()
  else:
    print("Program exited, because no suitable mode was defined. \
           The mode flag has to be set to one of the following:")
    print("  train")
    print("  decode")
    print("  generate_data")
    print("  filter_data")
    print("  experiment")


if __name__=="__main__":
  main()