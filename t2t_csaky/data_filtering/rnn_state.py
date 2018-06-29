
import os
import numpy as np
import faiss

# my imports
from . import filter_problem
from config import *

K = 3


class DataPoint(filter_problem.DataPoint):
  """
  A simple class that handles a string example.
  """
  def __init__(self, meaning_vector, string, index, only_string=True):
    super().__init__(string, index, only_string)
    self.meaning_vector = meaning_vector


class RNNState(filter_problem.FilterProblem):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  def DataPointClass(self):
    return DataPoint

  def clustering(self, data_tag):
    sentence_mv_path = self._data_path('{}States'.format(data_tag), 'npy')
    meaning_vectors = np.load(sentence_mv_path)

    centroids = calculate_centroids(meaning_vectors, K)
    clusters = [filter_problem.Cluster(centroid) for centroid in centroids]
    for data_point in self.data_points:
      cluster_index = calculate_nearest_index(
        data_point.meaning_vector, centroids)

      clusters[cluster_index].add_element(data_point)
      data_point.cluster_index = cluster_index

  # this function will read the data and make it ready for clustering
  def read_inputs(self):
    def read_sentences(file):
      sentences = []
      with open(file, 'r', encoding='utf-8') as f:
        for line in f:
          sentences.append(line.strip('\n'))
      return sentences

    def read(data_tag):
      # if the encodings exists they will not be generated again
      regular_data_path = self._data_path(data_tag, 'txt')
      ordered_data_path = self._data_path('{}Ordered'.format(data_tag), 'txt')
      # sentence meaning vector
      sentence_mv_path = self._data_path('{}States'.format(data_tag), 'npy')

      if (not os.path.exists(ordered_data_path) or
          not os.path.exists(sentence_mv_path)):
        generate_encoder_states(
          regular_data_path, ordered_data_path, sentence_mv_path)

      meaning_vectors = np.load(sentence_mv_path)

      sentence_dict = dict(zip(
        read_sentences(ordered_data_path), meaning_vectors))

      file = open(self._data_path(data_tag, 'txt'), 'r', encoding='utf-8')

      for index, line in enumerate(file):
        self.data_points[data_tag].append(self.DataPointClass(
          sentence_dict[line.strip('\n')], line, index, False))

      file.close()

    read('Source')
    read('Target')

    print("Finished reading " + self.tag + " data.")

  def _data_path(self, name, ext):
    return os.path.join(
        self.input_data_dir, self.tag + "{}.{}".format(name, ext))


def generate_encoder_states(regular_data_path,
                            ordered_data_path,
                            sentence_state_path):
  pass


def calculate_centroids(data_set, k):
  return []


def calculate_nearest_index(data_point, centroids):
  return 0
