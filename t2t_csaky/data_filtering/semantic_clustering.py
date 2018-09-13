
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import filter_problem
from config import DATA_FILTERING, FLAGS

from utils.utils import calculate_centroids_mean_shift
from utils.utils import calculate_centroids_kmeans

from sklearn.neighbors import BallTree


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
      centroids, method, = calculate_centroids_mean_shift(
        meaning_vectors)

    else:
      n_clusters = DATA_FILTERING['{}_clusters'.format(data_tag.lower())]
      centroids, method, = calculate_centroids_kmeans(
        meaning_vectors, niter=20, n_clusters=n_clusters)

    data_point_vectors = np.array([data_point.meaning_vector
                          for data_point in self.data_points[data_tag]])

    tree = BallTree(data_point_vectors)
    _, centroids = tree.query(centroids, k=1)
    tree = BallTree(centroids)
    _, labels = tree.query(data_point_vectors, k=1)

    clusters = [self.ClusterClass(
      self.data_points[data_tag][index])
     for index in {labels[_index] for _index in range(len(labels))}]

    rev_tag = "Target" if data_tag == "Source" else "Source"

    for data_point, cluster_index in zip(self.data_points[data_tag], labels):
      clusters[cluster_index].add_element(data_point)

      data_point.cluster_index = cluster_index

      clusters[cluster_index].targets.append(
        self.data_points[rev_tag][data_point.index])

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
      self.input_data_dir, "{}{}".format(name, ext))