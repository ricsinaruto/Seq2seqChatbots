import os
import argparse
import logging
import numpy
import scipy

import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.utils import split_sts_data
from utils.utils import tokenize_sentence
from utils.utils import calculate_correlation
from utils.utils import process_correlations


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
                            '-v', '/media/patrik/1EDB65B8599DD93E/GitHub/'
                                  'Seq2seqChatbots/data_dir/DailyDialog/'
                                  'base_both_avg_embedding_clustering/vocab'])

  assert os.path.isdir(args.output)

  file = os.path.splitext(os.path.basename(args.input))

  split_input_fst, split_input_snd = split_sts_data(
      args.input, file, args.output)

  # Collecting the existing words in the data, and records their frequency.
  word_count = 0
  vocab = {}
  with open(split_input_fst, 'r') as f:
    for line in f:
      for word in tokenize_sentence(line.strip().split()):
        vocab[word] = vocab.get(word, 0) + 1
        word_count += 1

  with open(split_input_snd, 'r') as f:
    for line in f:
      for word in tokenize_sentence(line.strip().split()):
        vocab[word] = vocab.get(word, 0) + 1
        word_count += 1

  for word in vocab:
    vocab[word] /= word_count

  # Iterating through the provided word vector vocabulary, and
  # pairing each word of the data with their frequency and embedding vector.
  dictionary = {}
  with open(args.vocab, 'r') as v:
    for line in v:
      line_as_list = line.strip().split()
      if line_as_list[0] in vocab:
        dictionary[line_as_list[0]] = (vocab[line_as_list[0]], numpy.array(
            [float(num) for num in line_as_list[1:]]))
        del vocab[line_as_list[0]]

  del vocab

  create_benchmark(args.input, dictionary)


def create_benchmark(sts_file_path, vocab):
  """
  Creates a benchmark for the provided sts file.
  Each sentence will be represented by the weighted average of the
  word embeddings in that sentence vectors.
  """

  # Inverse frequency weight.
  def w_avg(freq):
    return 0.001 / (0.001 + freq)

  target_correlation = []
  predicted_correlation = []
  with open(sts_file_path, 'r') as f:
    for line in f:
      line_as_list = line.split('\t')

      valid_words = 0
      vectors = []

      # STS data is a .csv, where the 5. and 6. columns hold the sentences,
      # and the 4. column holds the correlations
      first_sentence = None
      for word in tokenize_sentence(line_as_list[5].strip().split()):

        # Each sentence is split into words, and the vector corresponding
        # to each element will be weighted, and summed.
        vector = vocab.get(word)
        if vector is not None:
          vectors.append(vector[1] * w_avg(vector[0]))
          valid_words += 1

      if valid_words != 0:
        # If there were any words in the sentence, to find a vector for,
        # represent the sentence by the average of these vectors.
        first_sentence = numpy.sum(numpy.array(vectors), axis=0) / valid_words

      vectors = []
      valid_words = 0
      second_sentence = None
      for word in tokenize_sentence(line_as_list[6].strip().split()):
        vector = vocab.get(word)
        if vector is not None:
          vectors.append(vector[1] * w_avg(vector[0]))
          valid_words += 1

      if valid_words != 0:
        second_sentence = numpy.sum(numpy.array(vectors), axis=0) / valid_words

      if first_sentence is not None and second_sentence is not None:
        # If both vectors contain more than 0 words in the vocab,
        # calculate their cosine similarity.
        predicted_correlation.append(calculate_correlation(
            first_sentence,
            second_sentence))
        target_correlation.append(float(line_as_list[4].strip()))

  # The predicted similarity and the target similarity is compared
  # with pearson rang correlation.
  target_correlation = numpy.array(target_correlation)
  predicted_correlation = numpy.array(predicted_correlation).reshape(-1)
  predicted_correlation = process_correlations(predicted_correlation)

  corr, pvalue = scipy.stats.spearmanr(target_correlation,
                                       predicted_correlation)

  error = (numpy.sqrt(
           numpy.sum((target_correlation - predicted_correlation) ** 2)) /
           len(predicted_correlation))

  logger.info('Average embedding Correlation error (MSE): {}, '
              'Pearson correlation {}, pvalue {}'.format(error, corr, pvalue))


if __name__ == '__main__':
    main()
