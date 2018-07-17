
import numpy
import os
from sklearn.metrics.pairwise import cosine_similarity


def split_sts_data(input_file_path, file, output_dir):
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


def calculate_correlation(fst_vector, snd_vector):
  return cosine_similarity(fst_vector.reshape(1, -1),
                           snd_vector.reshape(1, -1))


def process_correlations(correlations):
  return (correlations - numpy.min(correlations)) / numpy.max(
    correlations) * 5