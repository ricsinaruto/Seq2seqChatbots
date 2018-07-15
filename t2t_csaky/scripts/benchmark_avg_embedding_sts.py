import os
import argparse
import logging
import numpy
import scipy
from sklearn.metrics.pairwise import cosine_similarity

import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

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
                            '-v', '/media/patrik/1EDB65B8599DD93E/'
                                  'Downloads/MUSE-master/data/'
                                  'wiki.en.clear.vec'])

  assert os.path.isdir(args.output)

  file = os.path.splitext(os.path.basename(args.input))

  # Used by the model for decoding output

  output_file_path_fst = os.path.join(
      args.output, '{}-first{}'.format(file[0], file[1]))

  output_file_path_snd = os.path.join(
      args.output, '{}-second{}'.format(file[0], file[1]))

  split_input_fst, split_input_snd = generate_input_data_for_model(
      args.input, file, args.output)

  vocab = set()
  with open(split_input_fst, 'r') as f:
    for line in f:
      for word in tokenize_sentence(line.strip().split()):
        vocab.add(word)

  with open(split_input_snd, 'r') as f:
    for line in f:
      for word in tokenize_sentence(line.strip().split()):
        vocab.add(word)

  dictionary = {}
  with open(args.vocab, 'r') as v:
    for line in v:
      line_as_list = line.strip().split()
      if line_as_list[0] in vocab:
        dictionary[line_as_list[0]] = numpy.array(
          [float(num) for num in line_as_list[-300:]])
        vocab.remove(line_as_list[0])

  del vocab

  create_benchmark(args.input, dictionary)


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
          o_snd.write(line_as_list[6].strip() + '\n')

  return split_input_path_fst, split_output_path_snd


def tokenize_sentence(line_as_list):
  tokenized_line = []
  for word in line_as_list:
    if word[-1] == '.':
      tokenized_line.append(word[:-1])
      tokenized_line.append('.')
    elif word[-1] == ',':
      tokenized_line.append(word[:-1])
      tokenized_line.append(',')
    else:
      tokenized_line.append(word)
  return tokenized_line


def create_benchmark(sts_file_path, vocab):
  target_correlation = []
  predicted_correlation = []
  with open(sts_file_path, 'r') as f:
    for line in f:
      line_as_list = line.split('\t')

      valid_words = 0
      vectors = []
      first_sentence = None
      for word in tokenize_sentence(line_as_list[5].strip().split()):
        vector = vocab.get(word)
        if vector is not None:
          vectors.append(vector)
          valid_words += 1

      if valid_words != 0:
        first_sentence = numpy.sum(numpy.array(vectors), axis=0) / valid_words

      vectors = []
      valid_words = 0
      second_sentence = None
      for word in tokenize_sentence(line_as_list[6].strip().split()):
        vector = vocab.get(word)
        if vector is not None:
          vectors.append(vector)
          valid_words += 1

      if valid_words != 0:
        second_sentence = numpy.sum(numpy.array(vectors), axis=0) / valid_words

      if first_sentence is not None and second_sentence is not None:
        predicted_correlation.append(calculate_correlation(
          first_sentence,
          second_sentence))
        target_correlation.append(float(line_as_list[4].strip()))

  target_correlation = numpy.array(target_correlation)

  predicted_correlation = numpy.array(predicted_correlation).reshape(-1)

  predicted_correlation = process_correlations(predicted_correlation)

  corr, pvalue = scipy.stats.spearmanr(target_correlation,
                                       predicted_correlation)

  error = numpy.sqrt(numpy.sum(
      (target_correlation - predicted_correlation) ** 2)) / \
          len(predicted_correlation)

  logger.info(
      'Average embedding Correlation error (MSE): {}, Pearson correlation {}, pvalue {}'.format(
          error, corr, pvalue))


def calculate_correlation(fst_vector, snd_vector):
    return cosine_similarity(fst_vector.reshape(1, -1),
                             snd_vector.reshape(1, -1))


def process_correlations(correlations):
    return (correlations - numpy.min(correlations)) / numpy.max(
        correlations) * 5


if __name__ == '__main__':
    main()
