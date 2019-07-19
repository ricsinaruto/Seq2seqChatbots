import argparse

from utils import run


def main():
  # Create an argument parser.
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, help='Can be one of the following: {\
                                                train,\
                                                decode,\
                                                generate_data,\
                                                experiment}')
  args = parser.parse_args()

  # Different modes will call different functions.
  run_mode = {
      'train': run.training,
      'decode': run.decoding,
      'generate_data': run.data_generating,
      'experiment': run.experiment
  }

  # Initialize a mode.
  if args.mode in run_mode:
    run_mode[args.mode]()
  else:
    print('Program exited, because no suitable mode was defined. \
           The mode flag has to be set to one of the following:')
    print('  train')
    print('  decode')
    print('  generate_data')
    print('  experiment')


if __name__ == '__main__':
  main()
