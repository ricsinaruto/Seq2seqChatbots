from datasketch import MinHash, MinHashLSHForest
import random

# my imports
from data_filtering.filter_problem import FilterProblem
from config import DATA_FILTERING


class DataPoint:
  """
  A class that handles a hash example.
  """
  def __init__(self, string, index, only_string=True):
    """
    Params:
      :string:  String to be stored.
      :index: Number of the line in the file from which this sentence was read.
      :only_string: Whether to only store string.
    """
    self.string = string.strip("\n")
    self.index = index
    self.character_level = DATA_FILTERING["character_level"]
    self.cluster_index = 0

    if not only_string:
      self.init_hash()

  # Initialize hash from string.
  def init_hash(self):
    self.min_hash = MinHash(num_perm=DATA_FILTERING["num_permutations"])
    for word in self.string.split():
      if self.character_level:
        for char in word:
          self.min_hash.update(char.encode('utf8'))
      else:
        self.min_hash.update(word.encode('utf8'))

  # Computes jaccard distance between self and another hash.
  def similarity(self, other, dist_matrix=""):
    return self.min_hash.jaccard(other.min_hash)


class HashJaccard(FilterProblem):
  """
  A class that does clustering based on hashes from the datasketch library.
  """
  @property
  def num_perm(self):
    return DATA_FILTERING["num_permutations"]

  @property
  def DataPointClass(self):
    return DataPoint

  # Find nearest medoid for a data point.
  def find_nearest_medoid(self, data_point, data_tag=""):
    nearest_medoid = self.forest.query(data_point.min_hash, 1)
    if not nearest_medoid:
      nearest_medoid = [random.randint(0, self.num_clusters[data_tag] - 1)]
    return nearest_medoid[0]

  # Do the clustering of sources and targets.
  def clustering(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data.
    """

    # Create a min hash forest to quickly find nearest neighbours.
    self.forest = MinHashLSHForest(num_perm=self.num_perm)

    # Initialize clusters.
    medoids = random.sample(range(len(self.data_points[data_tag])),
                            self.num_clusters[data_tag])

    for i in range(self.num_clusters[data_tag]):
      cl = self.ClusterClass(self.data_points[data_tag][medoids[i]])
      self.clusters[data_tag].append(cl)

      # Put medoids in a the forest.
      self.forest.add(i, self.clusters[data_tag][-1].medoid.min_hash)
    self.forest.index()

    # For each data_point find a cluster.
    self.cluster_points(data_tag)

    # These will be needed for the stopping criterion.
    cluster_names = [self.clusters[data_tag][i].medoid.string
                     for i in range(self.num_clusters[data_tag])]
    cluster_names_old = list(cluster_names)
    count = 0
    counts = []
    exit = False

    # Clustering loop.
    while not exit:
      count += 1

      # Find the point that minimizes the mean distance within a cluster.
      self.find_medoid(data_tag)

      # Create new forest.
      self.forest = MinHashLSHForest(num_perm=self.num_perm)
      for i in range(self.num_clusters[data_tag]):
        self.forest.add(i, self.clusters[data_tag][i].medoid.min_hash)
      self.forest.index()

      # Assign each point to the new medoids.
      self.cluster_points(data_tag)

      # Check stopping criterions.
      exit, cluster_names, cluster_names_old, counts = self.stop_clustering(
          data_tag,
          cluster_names,
          cluster_names_old,
          count,
          counts)
