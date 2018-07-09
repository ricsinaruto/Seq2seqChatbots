
import os
import argparse
import pandas


import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import FLAGS


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str)
  parser.add_argument('-o', '--output', type=str)
  parser.add_argument('-v', '--vocab', type=str)

  args = parser.parse_args(['-i', '/media/patrik/1EDB65B8599DD93E/Downloads'
                                  '/stsbenchmark/sts-train.csv',
                            '-o', '/media/patrik/1EDB65B8599DD93E/GitHub'
                                  '/Seq2seqChatbots/sts_benchmark',
                            '-v', '/media/patrik/1EDB65B8599DD93E'
                                  '/GitHub/Seq2seqChatbots/data_dir'
                                  '/DailyDialog/base_with_numbers'
                                  '/vocab.chatbot.16384'])

  assert os.path.isdir(args.output)

  file = os.path.splitext(os.path.basename(args.input))

  # Used by the model for decoding output

  output_file_path_fst = os.path.join(
    args.output, '{}-first{}'.format(file[0], file[1]))

  output_file_path_snd = os.path.join(
    args.output, '{}-second{}'.format(file[0], file[1]))

  with open(args.vocab, 'r', encoding='utf-8') as v:
    vocab = {line.strip() for line in v if line.strip() != ''}

  temp_fst = os.path.join(
    args.output, '{}-first-temp{}'.format(file[0], file[1]))

  temp_snd = os.path.join(
    args.output, '{}-second-temp{}'.format(file[0], file[1]))

  split_input_fst, split_input_snd = generate_input_data_for_model(
    args.input, file, args.output)

  fst_sentence_dict = {}

  with open(temp_fst, 'w') as temp_f:
    with open(split_input_fst, 'r', encoding='utf-8') as f:
      for line in f:
        reduced_sentence = ' '.join([
          word for word in line.strip().split() if word in vocab])
        fst_sentence_dict[reduced_sentence] = line.strip()
        temp_f.write(reduced_sentence + '\n')

  generate_states(temp_fst, output_file_path_fst)
  os.remove(temp_fst)

  transformed_output = []
  with open(output_file_path_fst, 'r', encoding='utf-8') as f:
    for line in f:
      transformed_output.append(line.strip())

  with open(output_file_path_fst, 'w', encoding='utf-8') as f:
    for line in transformed_output:
      f.write(fst_sentence_dict[line] + '\n')

  del fst_sentence_dict

  snd_sentence_dict = {}

  with open(temp_snd, 'w') as temp_f:
    with open(split_input_fst, 'r', encoding='utf-8') as f:
      for line in f:
        reduced_sentence = ' '.join([
          word for word in line.strip().split() if word in vocab])
        snd_sentence_dict[reduced_sentence] = line.strip()
        temp_f.write(reduced_sentence + '\n')

  generate_states(temp_snd, output_file_path_snd)
  os.remove(temp_snd)

  transformed_output = []
  with open(output_file_path_snd, 'r', encoding='utf-8') as f:
    for line in f:
      transformed_output.append(line.strip())

  with open(output_file_path_snd, 'w', encoding='utf-8') as f:
    for line in transformed_output:
      f.write(snd_sentence_dict[line] + '\n')

  os.remove(split_input_fst)
  os.remove(split_input_snd)

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

  split_input_path_fst = os.path.join(
    output_dir, '{}-first-split{}'.format(file[0], file[1]))

  split_output_path_snd = os.path.join(
    output_dir, '{}-second-split{}'.format(file[0], file[1]))


  with open(input_file_path, 'r', encoding='utf-8') as i_f:
    with open(split_input_path_fst, 'w', encoding='utf-8') as o_fst:
      with open(split_output_path_snd, 'w', encoding='utf-8') as o_snd:
        for line in i_f:
          line_as_list = line.strip().split('\t')
          o_fst.write(line_as_list[5].strip() + '\n')
          o_snd.write(line_as_list[6].strip())

  return split_input_path_fst, split_output_path_snd


if __name__ == '__main__':
    main()
