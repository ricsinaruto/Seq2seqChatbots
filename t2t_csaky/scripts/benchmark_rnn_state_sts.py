
import os
import argparse
import logging
import numpy
import scipy

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from scripts.utils import split_sts_data
from scripts.utils import tokenize_sentence
from scripts.utils import calculate_correlation
from scripts.utils import process_correlations


from config import FLAGS


logFormatter = logging.Formatter(
    "%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

fileHandler = logging.FileHandler(
  os.path.join(os.path.dirname(os.path.abspath(__file__)),
               '..', '..', 'sts_benchmark', 'benchmark.log'))
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


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

  split_input_fst, split_input_snd = split_sts_data(
    args.input, file, args.output)

  fst_sentence_dict = {}

  with open(temp_fst, 'w') as temp_f:
    with open(split_input_fst, 'r', encoding='utf-8') as f:
      for line in f:
        reduced_sentence = ' '.join([
          word for word in tokenize_sentence(line.strip().split())
          if word in vocab and word.strip() != ''])
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

  snd_sentence_dict = {}

  with open(temp_snd, 'w') as temp_f:
    with open(split_input_snd, 'r', encoding='utf-8') as f:
      for line in f:
        reduced_sentence = ' '.join([
          word for word in tokenize_sentence(line.strip().split()) if
          word in vocab and word.strip() != ''])
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

  fst_dict, snd_dict = create_sentence_dicts(
    output_file_path_fst,
    output_file_path_snd,
    os.path.splitext(output_file_path_fst)[0] + '.npy',
    os.path.splitext(output_file_path_snd)[0] + '.npy',
    vocab)

  create_benchmark(args.input, fst_dict, snd_dict,vocab)

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


def create_sentence_dicts(fst_split_csv_path,
                          snd_split_csv_path,
                          fst_split_npy_path,
                          snd_split_npy_path,
                          vocab):

  fst_sentence_dict = {}
  sentence_states = numpy.load(fst_split_npy_path)

  with open(fst_split_csv_path, 'r') as f:
    for index, line in enumerate(f):
      fst_sentence_dict[' '.join(
        [word for word in tokenize_sentence(
          line.strip().split()) if word in vocab])] = \
        sentence_states[index]

  del sentence_states

  snd_sentence_dict = {}
  sentence_states = numpy.load(snd_split_npy_path)

  with open(snd_split_csv_path, 'r') as f:
    for index, line in enumerate(f):
      snd_sentence_dict[' '.join(
        [word for word in tokenize_sentence(
          line.strip().split()) if word in vocab])]\
        = sentence_states[index]

  del sentence_states

  return fst_sentence_dict, snd_sentence_dict


def create_benchmark(sts_file_path, fst_dict, snd_dict, vocab):
  target_correlation = []
  predicted_correlation =[]
  with open(sts_file_path, 'r') as f:
    for line in f:
      line_as_list = line.split('\t')
      first_sentence = \
        [word for word in tokenize_sentence(line_as_list[5].strip().split())
                             if word in vocab and word != '']
      second_sentence = \
        [word for word in tokenize_sentence(line_as_list[6].strip().split())
                             if word in vocab and word != '']
      if len(first_sentence) > 2 and len(second_sentence) > 2:
        predicted_correlation.append(calculate_correlation(
          fst_dict[' '.join(first_sentence)],
          snd_dict[' '.join(second_sentence)]))
        target_correlation.append(float(line_as_list[4].strip()))

  target_correlation = numpy.array(target_correlation)
  predicted_correlation = numpy.array(predicted_correlation).reshape(-1)
  predicted_correlation = process_correlations(predicted_correlation)

  corr, pvalue = scipy.stats.spearmanr(target_correlation,
                                       predicted_correlation)

  error = numpy.sqrt(numpy.sum(
    (target_correlation - predicted_correlation) ** 2)) / \
          len(predicted_correlation)

  logger.info('RNNState Correlation error (MSE): {}, '
              'Pearson correlation {}, pvalue {}'.format(error, corr, pvalue))


if __name__ == '__main__':
    main()
