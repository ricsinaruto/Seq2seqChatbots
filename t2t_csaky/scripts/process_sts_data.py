
import os
import argparse

from t2t_csaky.config import FLAGS


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str)
  parser.add_argument('-o', '--output', type=str)

  args = parser.parse_args()

  assert os.path.isdir(args.output)

  file = os.path.split(os.path.basename(args.input))

  output_file_path_fst = os.path.join(
    args.output, '{}-first{}'.format(file[0], file[1]))

  output_file_path_snd = os.path.join(
    args.output, '{}-second{}'.format(file[0], file[1]))


def generate_states(input_file_path, output_file_path):

  # what hparams should we use
  if FLAGS["hparams"] == "":
    hparam_string = "general_" + FLAGS["model"] + "_hparams"
  else:
    hparam_string = FLAGS["hparams"]

  decode_mode_string = ""
  # determine the decode mode flag
  if FLAGS["decode_mode"] == "interactive":
    decode_mode_string = " --decode_interactive"
  elif FLAGS["decode_mode"] == "file":
    decode_mode_string = (" --decode_from_file=" + input_file_path)

  script_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      'state_extraction.py')

  os.system("python3 {} \
                  --generate_data=False \
                  --t2t_usr_dir=".format(script_path) + FLAGS["t2t_usr_dir"]
            + " --data_dir=" + FLAGS["data_dir"]
            + " --problem=" + FLAGS["problem"]
            + " --output_dir=" + FLAGS["train_dir"]
            + " --model=" + FLAGS["model"]
            + " --worker_gpu_memory_fraction=" + str(FLAGS["memory_fraction"])
            + " --hparams_set=" + hparam_string
            + " --decode_to_file=" + output_file_path
            + ' --decode_hparams="beam_size=' + str(FLAGS["beam_size"])
            + ",return_beams=" + FLAGS["return_beams"] + '"'
            + decode_mode_string)


def generate_input_data_for_model(input_file_path, file, output_dir):

  temp_output_file_path_fst = os.path.join(
    output_dir, '{}-first-tmp{}'.format(file[0], file[1]))

  temp_output_file_path_snd = os.path.join(
    output_dir, '{}-second-tmp{}'.format(file[0], file[1]))

  with open(input_file_path, 'r', encoding='utf-8') as i_f:
    with open(temp_output_file_path_fst, 'w', encoding='utf-8') as o_fst:
      with open(temp_output_file_path_snd, 'w', encoding='utf-8') as o_snd:
        for line in i_f:
          line_as_list = line.strip().split('\t')
          o_fst.write(line_as_list[-2].strip() + '\n')
          o_snd.write(line_as_list[-1].strip())

  return temp_output_file_path_fst, temp_output_file_path_snd


if __name__ == '__main__':
    main()
