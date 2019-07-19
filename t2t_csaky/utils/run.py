import os
import datetime
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import FLAGS


# Save the config.py file for a specific run.
def save_config_file(directory):
  # Make the data dir if it doesn't exist.
  if not os.path.exists(directory):
    os.makedirs(directory)

  # This will be used in the names of saved files.
  now = datetime.datetime.now()
  time_string = (str(now.year) + '.' +
                 str(now.month) + '.' +
                 str(now.day) + '.' +
                 str(now.hour) + '.' +
                 str(now.minute) + '.' +
                 str(now.second))

  os.system('cp ' + FLAGS['t2t_usr_dir'] + '/config.py ' +
            directory + '/config.' + time_string + '.txt')


# Initialize a data generation problem.
def data_generating():
  print('Program is running in data generation mode.')
  save_config_file(FLAGS['data_dir'])
  os.system('t2t-datagen \
               --t2t_usr_dir=' + FLAGS['t2t_usr_dir'] +
            ' --data_dir=' + FLAGS['data_dir'] +
            ' --problem=' + FLAGS['problem'])


# initialize a training loop with the given flags.
def training():
  print('Program is running in training mode.')
  save_config_file(FLAGS['train_dir'])

  # What hparams should we use.
  if FLAGS['hparams'] == '':
    hparam_string = 'general_' + FLAGS['model'] + '_hparams'
  else:
    hparam_string = FLAGS['hparams']

  os.system('t2t-trainer \
               --generate_data=False \
               --t2t_usr_dir=' + FLAGS['t2t_usr_dir'] +
            ' --data_dir=' + FLAGS['data_dir'] +
            ' --problem=' + FLAGS['problem'] +
            ' --output_dir=' + FLAGS['train_dir'] +
            ' --model=' + FLAGS['model'] +
            ' --hparams_set=' + hparam_string +
            ' --schedule=' + FLAGS['train_mode'] +
            ' --worker_gpu_memory_fraction=' + str(FLAGS['memory_fraction']) +
            ' --keep_checkpoint_max=' + str(FLAGS['keep_checkpoints']) +
            ' --keep_checkpoint_every_n_hours=' +
            str(FLAGS['save_every_n_hour']) +
            ' --save_checkpoints_secs=' + str(FLAGS['save_every_n_secs']) +
            ' --train_steps=' + str(FLAGS['train_steps']) +
            ' --eval_steps=' + str(FLAGS['evaluation_steps']) +
            ' --local_eval_frequency=' + str(FLAGS['evaluation_freq']))


# Intialize an inference test with the given flags.
def decoding():
  print('Program is running in inference/decoding mode.')
  save_config_file(FLAGS['decode_dir'])

  # What hparams should we use.
  if FLAGS['hparams'] == '':
    hparam_string = 'general_' + FLAGS['model'] + '_hparams'
  else:
    hparam_string = FLAGS['hparams']

  decode_mode_string = ''
  # Determine the decode mode flag.
  if FLAGS['decode_mode'] == 'interactive':
    decode_mode_string = ' --decode_interactive'
  elif FLAGS['decode_mode'] == 'file':
    decode_mode_string = (' --decode_from_file=' +
                          FLAGS['decode_dir'] + '/' +
                          FLAGS['input_file_name'])

  os.system('t2t-decoder \
               --generate_data=False \
               --t2t_usr_dir=' + FLAGS['t2t_usr_dir'] +
            ' --data_dir=' + FLAGS['data_dir'] +
            ' --problem=' + FLAGS['problem'] +
            ' --output_dir=' + FLAGS['train_dir'] +
            ' --model=' + FLAGS['model'] +
            ' --worker_gpu_memory_fraction=' + str(FLAGS['memory_fraction']) +
            ' --hparams_set=' + hparam_string +
            ' --decode_to_file=' +
            FLAGS['decode_dir'] + '/' + FLAGS['output_file_name'] +
            ' --decode_hparams=\'beam_size=' + str(FLAGS['beam_size']) +
            ',return_beams=' + FLAGS['return_beams'] +
            ',batch_size=' + str(FLAGS['batch_size']) + '\'' +
            decode_mode_string)


# Run a longer experiment, with many calls to the above functions.
def experiment():
  pass
