
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

import filter_problem
from config import DATA_FILTERING, FLAGS

_use_faiss = False

if DATA_FILTERING['use_faiss']:
  try:

    import faiss
    _use_faiss = True

  except ImportError:
    print('Failed to import faiss, using SKLearn clustering instead.')

if not _use_faiss:
  from sklearn.cluster import KMeans

from sklearn.cluster import MeanShift


class DataPoint(filter_problem.DataPoint):
  """
  A simple class that handles a string example.
  """
  def __init__(self, string, index, only_string=True, meaning_vector=None):
    """
    Params:
      :string:  String to be stored
      :index: Number of the line in the file from which this sentence was read
      :only_string: Whether to only store string
      :meaning_vector: Numpy embedding vector for the sentence
    """
    super().__init__(string, index, only_string)
    self.meaning_vector = meaning_vector


class SemanticClustering(filter_problem.FilterProblem):
  """
  Base class for the meaning-based (semantic vector representation) clustering.
  The source and target sentences are read into an extended DataPoint object,
  that also contains a 'meaning_vector' attribute. This attribute holds
  the semantic vector representation of the sentence, which will be used
  by the clustering logic.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.paths = {}

  @staticmethod
  def simple_knn(data_point, data_set):
    """
    Finds the index of the nearest vector to the provided vector.

    Params:
      :data_point: A single data point, to find the nearest index for.
      :data_set: A set of data points.

    Returns:
      Index of the nearest neighbour for the provided vector.
    """
    return np.argmin(np.sum((data_set - data_point) ** 2, 1))

  @staticmethod
  def calculate_centroids_kmeans(data_set, niter):
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
      kmeans = faiss.Kmeans(d, DATA_FILTERING['kmeans_n_clusters'],
                            niter, verbose)
      kmeans.train(data_set)
      centroids = kmeans.centroids

    else:
      kmeans = KMeans(n_clusters=DATA_FILTERING['kmeans_n_clusters'],
                      random_state=0,
                      n_jobs=10).fit(data_set)

      centroids = kmeans.cluster_centers_

    return centroids, kmeans

  @staticmethod
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

  @staticmethod
  def calculate_nearest_index(data, method):
    """
    Calculates the cluster centroid for the provided vector.
    """
    if _use_faiss:
      _, index = method.index.search(data, 1)

    else:
      index = method.predict(data)[0]

    return index

  @staticmethod
  def read_sentences(file):
    """
    Convenience method for reading the sentences of a file.
    """
    sentences = []
    with open(file, 'r', encoding='utf-8') as f:
      for line in f:
        sentences.append(' '.join(
          [word for word in line.strip().split() if word.strip() != ''
           and word.strip() != '<unk>']))
    return sentences

  @property
  def DataPointClass(self):
    return DataPoint

  def clustering(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """
    meaning_vectors = np.load(self.paths[data_tag]['npy'])

    if DATA_FILTERING["semantic_clustering_method"] == "mean_shift":
      centroids, method = self.calculate_centroids_mean_shift(
        meaning_vectors, )

    else:
      centroids, method = self.calculate_centroids_kmeans(
        meaning_vectors, niter=20)

    data_point_vectors = np.array([data_point.meaning_vector
                          for data_point in self.data_points[data_tag]])

    clusters = [self.ClusterClass(
      self.data_points[data_tag][self.simple_knn(
        centroid, data_point_vectors)])
     for centroid in centroids]

    rev_tag = "Target" if data_tag == "Source" else "Source"

    for data_point in self.data_points[data_tag]:
      cluster_index = self.calculate_nearest_index(
        data_point.meaning_vector.reshape(1, -1), method)
      clusters[cluster_index].add_element(data_point)
      data_point.cluster_index = cluster_index
      clusters[cluster_index]\
        .targets.append(self.data_points[rev_tag][data_point.index])

    self.clusters[data_tag] = clusters

  def read_inputs(self):
    """
    This function will read the data and make it ready for clustering.
    """
    self._read('Source')
    self._read('Target')

    print("Finished reading " + self.tag + " data.")

  def _read(self, data_tag):
    """
    This function will be called twice by the read_inputs method,
    with 'Source' and 'Target' data_tags. It should implement the
    logic of reading the data from Source and Target files into the
    data_points list. Each sentence should be wrapped into an appropriate
    subclass of the DataPoint class.
    """
    raise NotImplementedError

  def _data_path(self, name, ext=''):
    """
    Convenience method for creating paths to the input data directory.

    Params:
      name: Name of the file
      :ext: Extension of the file
    """
    return os.path.join(
      self.input_data_dir, self.tag + "{}{}".format(name, ext))