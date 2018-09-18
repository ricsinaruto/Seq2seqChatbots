import os
import sys
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MeanShift

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

# My imports.
from config import DATA_FILTERING, FLAGS, PROBLEM_HPARAMS

_use_faiss = False

if DATA_FILTERING['use_faiss']:
  try:
    import faiss
    _use_faiss = True

  except ImportError:
    print('Failed to import faiss, using SKLearn clustering instead.')

if not _use_faiss:
  from sklearn.cluster import KMeans


# Temporary helper function to load a vocabulary.
def load_vocab():
  vocab = open(os.path.join(FLAGS["data_dir"],
               "vocab.chatbot." + str(PROBLEM_HPARAMS["vocabulary_size"])))
  vocab_dict = {}
  # Read the vocab file.
  i = 0
  for word in vocab:
    vocab_dict[word.strip("\n")] = i
    i += 1

  vocab.close()
  return vocab_dict


def split_sts_data(input_file_path, file, output_dir):
  """
  Convenience function that is used exclusively for processsing
  the STS benchmark data. The file contains sentences-pairs,
  which will be split into two separate files.
  """
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
  """
  Tokenizes the sentence by separating punctuation marks at the end of
  each word.
  """
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
  """
  Calcualtes the cosine similarity of two vectors for STS benchmarking.
  """
  return cosine_similarity(fst_vector.reshape(1, -1),
                           snd_vector.reshape(1, -1))


def process_correlations(correlations):
  """
  Rescales the vectors into a 0-5 interval.
  """
  return (correlations - numpy.min(correlations)) / numpy.max(correlations) * 5


def simple_knn(data_point, data_set):
  """
  Finds the index of the nearest vector to the provided vector.

  Params:
    :data_point: A single data point, to find the nearest index for.
    :data_set: A set of data points.

  Returns:
    Index of the nearest neighbour for the provided vector.
  """
  return numpy.argmin(numpy.sum((data_set - data_point) ** 2, 1))


def calculate_centroids_kmeans(data_set, niter, n_clusters):
  """
  Clusters the provided data set with kmeans with either the faiss or
  the sklearn implementation.

  Params:
    :data_set: A set of vectors.
    :niter: Number of max iterations.
  """
  if _use_faiss:
    verbose = True
    d = data_set.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter, verbose)
    kmeans.train(data_set)
    centroids = kmeans.centroids

  else:
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=0,
                    n_jobs=10).fit(data_set)

    centroids = kmeans.cluster_centers_

  return centroids, kmeans


def calculate_centroids_mean_shift(data_set):
  """
  Clusters the provided dataset, using mean shift clustering.

  Params:
    :data_set: A set of vectors.
  """
  mean_shift = MeanShift(
      bandwidth=DATA_FILTERING['mean_shift_bw']).fit(data_set)
  centroids = mean_shift.cluster_centers_

  return centroids, mean_shift


def calculate_nearest_index(data, method):
  """
  Calculates the cluster centroid for the provided vector.
  """
  if _use_faiss:
    _, index = method.index.search(data, 1)
  else:
    index = method.predict(data)[0]

  return index


def read_sentences(file):
  """
  Convenience method for reading the sentences of a file.
  """
  sentences = []
  with open(file, 'r', encoding='utf-8') as f:
    for line in f:
      sentences.append(' '.join(
          [word for word in line.strip().split() if
           word.strip() != '' and word.strip() != '<unk>']))
  return sentences
