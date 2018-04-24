from datasketch import MinHash, MinHashLSH, MinHashLSHForest
import random

# my imports
from data_filtering.filter_problem import FilterProblem
from config import *



class DataPoint:
  """
  A class that handles a hash example.
  """
  def __init__(self, string, only_string=True):
    """
    Params:
      :string:  String to be stored
      :only_string: Whether to only store string
    """ 
    self.string=string.strip("\n")
    self.character_level=DATA_FILTERING["character_level"]
    self.cluster_index=0

    if not only_string:
      self.init_hash()

  # initialize hash from string
  def init_hash(self):
    self.min_hash=MinHash(num_perm=DATA_FILTERING["num_permutations"])
    for word in self.string.split():
      if self.character_level:
        for char in word:
          self.min_hash.update(char.encode('utf8'))
      else:
        self.min_hash.update(word.encode('utf8'))

  # computes jaccard distance between self and another hash
  def distance(self, other):
    return self.min_hash.jaccard(other.min_hash)


class HashJaccard(FilterProblem):
  """
  A class that does clustering based on hashes provided by the datasketch library.
  """
  @property
  def num_perm(self):
    return DATA_FILTERING["num_perm_forest"]

  @property
  def DataPointClass(self):
    return DataPoint

  # find nearest medoid for a data point
  def find_nearest_medoid(self, data_point):
    nearest_medoid=self.forest.query(data_point.min_hash, 1)
    if not nearest_medoid:
      nearest_medoid=[random.randint(0, self.num_clusters-1)]
    return nearest_medoid[0]

  # do the clustering of sources and targets
  def clustering(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """

    # create a min hash forest to quickly find nearest neighbours
    self.forest=MinHashLSHForest(num_perm=self.num_perm)

    # initialize clusters
    medoids=random.sample(range(len(self.data_points[data_tag])), self.num_clusters)
    for i in range(self.num_clusters):
      self.clusters[data_tag].append(self.ClusterClass(self.data_points[data_tag][medoids[i]]))
      # put medoids in a the forest
      self.forest.add(i, self.clusters[data_tag][-1].medoid.min_hash)
    self.forest.index()

    # for each data_point find a cluster
    self.cluster_points(data_tag)
          
    # these will be needed for the stopping criterion
    cluster_names=[self.clusters[data_tag][i].medoid.string for i in range(self.num_clusters)]
    cluster_names_old=list(cluster_names)
    loop_count=0
    exit=False

    # clustering loop
    while not exit:
      loop_count+=1

      # find the point that minimizes the mean distance within a cluster
      self.find_medoid(data_tag)

      # create new forest
      self.forest=MinHashLSHForest(num_perm=self.num_perm)
      for i in range(self.num_clusters):
        self.forest.add(i, self.clusters[data_tag][i].medoid.min_hash)
      self.forest.index()

      # assign each point to the new medoids
      self.cluster_points(data_tag)

      # check stopping criterions
      exit, cluster_names, cluster_names_old=self.stop_clustering(data_tag, cluster_names, cluster_names_old, loop_count)

    # save the clustering results
    self.save_clusters(data_tag)