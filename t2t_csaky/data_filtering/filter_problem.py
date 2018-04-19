import os

# my imports
from hash_jaccard import HashJaccard
from sentence_embedding import SentenceEmbedding
from rnn_state import RNNState
from ..config import *


# dictionary to choose between the different filter problem classes
filter_problems={
  "hash_jaccard"      :HashJaccard()
  "sentence_embedding":SentenceEmbedding()
  "rnn_state"         :RNNState()
}



class DataPoint:
  """
  A simple class that handles a string example.
  """
  def __init__(self, string):
    self.string=string.strip("\n")
    self.cluster_index=0


class Cluster:
  """
  A class to handle one cluster in the clustering problem.
  """
  def __init__(self, medoid):
    """
    Params:
      :medoid:  Center of the cluster.
    """
    self.medoid=medoid
    self.elements=[]
    self.targets=[]

  # append an element to the list of elements in the cluster
  def add_element(self, element):
    self.elements.append(element)


class FilterProblem:
  """
  An abstract class to handle filtering of different types.
  """

  @property
  def DataPointClass(self):
    return DataPoint

  @property
  def ClusterClass(self):
    return Cluster

  def __init__(self, tag="train"):
    """
    Params:
      :tag: Can be either train, dev or test
    """
    self.tag=tag
    self.treshold=DATA_FILTERING["treshold"]
    self.clusters= {
      "Source" : []
      "Target" : []
    }
    self.data_points= {
      "Source" : []
      "Target" : []
    }

    # calculate number of clusters
    if self.tag=="train":
      self.num_clusters=DATA_FILTERING["num_clusters"]
    elif self.tag=="dev":
      self.num_clusters=int(DATA_FILTERING["num_clusters"]
                            * PROBLEM_HPARAMS["dataset_split"]["val"]
                            / PROBLEM_HPARAMS["dataset_split"]["train"])
    else:
      self.num_clusters=int(DATA_FILTERING["num_clusters"]
                            * PROBLEM_HPARAMS["dataset_split"]["test"]
                            / PROBLEM_HPARAMS["dataset_split"]["train"])

    self.output_data_dir=DATA_FILTERING["data_dir"]
    self.input_data_dir=FLAGS["data_dir"]
    self.type=DATA_FILTERING["filter_type"]

  # main method that will run all the functions to do the filtering
  def run(self):
    self.read_inputs()

    # if we have already done the clustering, don't redo it
    if os.path.isfile(os.path.join(self.output_data_dir, self.tag+"Source_cluster_elements.txt")) \
      and os.parh.isfile(os.path.join(self.output_data_dir, self.tag+"Target_cluster_elements.txt")):

      print("Found cluster files in "+self.output_data_dir+", proceeding with filtering.")
      self.load_clusters("Source")
      self.load_clusters("Target")
      self.filtering()

    else:
      print("No cluster files were found in "+self.output_data_dir+", proceeding with clustering.")
      self.clustering("Source")
      self.clustering("Target")
      self.filtering()

  # this function will read the data and make it ready for clustering
  def read_inputs(self):
    sources=open(os.path.join(self.input_data_dir, self.tag+"Source"))
    targets=open(os.path.join(self.input_data_dir, self.tag+"Target"))

    for line in sources:
      data_points["Source"].append(self.DataPointClass(line))

    for line in targets:
      data_points["Target"].append(self.DataPointClass(line))

    sources.close()
    targets.close()
    print("Finished reading "+self.tag+" data.")

  # load clusters from files
  def load_clusters(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """
    clusters=open(os.path.join(self.output_data_dir, self.tag+data_tag+"_cluster_elements.txt"))

    # temporary dict to facilitate loading clusters
    temp_dict={}
    for i in range(self.data_points[data_tag]):
      temp_dict[self.data_points[data_tag][i].string]=i

    # load the clusters
    last_medoid="<-->"
    for line in clusters:
      [medoid, element]=line.strip("\n").split("<=====>")

      # check if this is a new medoid
      if last_medoid=="<-->" or last_medoid!=medoid:
        # add medoid to cluster
        self.clusters[data_tag].append(self.ClusterClass(data_points[data_tag][temp_dict[medoid]]))
      self.clusters[data_tag][-1].add_element(data_points[data_tag][temp_dict[element]])

      # reverse data tag
      if data_tag=="Source":
        self.clusters[data_tag][-1].targets.append(data_points["Target"][temp_dict[element]])
      else:
        self.clusters[data_tag][-1].targets.append(data_points["Source"][temp_dict[element]])

      # add cluster index to data points
      self.data_points[data_tag][temp_dict[element]].cluster_index=len(self.clusters[data_tag])-1
      last_medoid=medoid

    clusters.close()

  # find the point that minimizes mean distance within a cluster
  def find_medoid(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """
    for i in range(self.num_clusters):
      big_sum=0
      for j1 in range(len(self.clusters[data_tag][i].elements)):
        small_sum=0
        for j2 in range(len(self.clusters[data_tag][i].elements)):
          small_sum+=self.clusters[data_tag][i].elements[j1].distance(self.clusters[data_tag][i].elements[j2])

        if small_sum>big_sum:
          big_sum=small_sum
          self.clusters[data_tag][i].medoid=self.clusters[data_tag][i].elements[j1]

      # clear elements after we finished with one cluster
      self.clusters[data_tag][i].elements.clear()
      self.clusters[data_tag][i].targets.clear()

  # find nearest medoid for a data point
  def find_nearest_medoid(self, data_point):
    NotImplementedError

  # for each data_point find a cluster
  def cluster_points(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """
    for i in range(len(self.data_points[data_tag])):
      nearest_medoid=self.find_nearest_medoid(self.data_points[data_tag][i])
      self.clusters[data_tag][nearest_medoid].add_element(self.data_points[data_tag][i])
      # reverse data tag
      if data_tag=="Source":
        self.clusters[data_tag][-1].targets.append(data_points["Target"][i])
      else:
        self.clusters[data_tag][-1].targets.append(data_points["Source"][i])

      self.data_points[data_tag][i].cluster_index=nearest_medoid

  def stop_clustering(self, data_tag, cluster_names, cluster_names_old, loop_count):
    """
    Params:
      :data_tag: Whether it's source or target data
      :cluster_names: String of medoids from previous iteration
      :cluster_names_old: String of medoids from 2 iterations ago
      :loop_count: Number of clustering loops so far
    """ 
    count_difference=0
    count_difference_old=0
    for i in range(self.num_clusters):
      # check strings from previous iteration to see if they are the same
      if self.clusters[data_tag][i].medoid.string!=cluster_names[i]:
        count_difference+=1
        print(cluster_names[i]+"-------->"+self.clusters[data_tag][i].medoid.string)
        cluster_names[i]=self.clusters[data_tag][i].medoid.string

      # check strings from two loops ago, to see if they are the same
      if self.clusters[data_tag][i].medoid.string!=cluster_names_old[i]:
        count_difference_old+=1
        if loop_count %2==0:
          cluster_names_old[i]=self.clusters[data_tag][i].medoid.string
    print("==================================================")
    print("==================================================")

    # exit if there is no change or we are stuck in a loop
    exit=True if count_difference==0 or count_difference_old==0 else False
    return exit, cluster_names, cluster_names_old

  # do the clustering of sources and targets
  def clustering(self, data_tag):
    return NotImplementedError

  # do the filtering of the dataset
  def filtering(self):
    if self.type=="target_based":
      self.get_filtered_indices("Source", "Target")


  # save clusters and their elements to files
  def save_clusters(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """
    output=open(os.path.join(self.output_data_dir, self.tag+data_tag+"_cluster_elements.txt"), "w")

    medoid_counts=[]
    for cluster in self.clusters[data_tag]:
      medoid_counts.append(cluster.medoid.string, len(cluster.elements))
      for data_point in cluster.elements:
        output.write(cluster.medoid.string+"<=====>"+data_point.string+"\n")
    output.close()

    # save the medoids and the count of their elements
    output=open(os.path.join(self.output_data_dir, self.tag+data_tag+"_clusters.txt"), "w")
    sorted_medoids=sorted(medoid_counts, key=lambda count: count[1], reverse=True)

    for medoid in medoids:
      output.write(medoid[0]+":"+str(medoid[1])+"\n")

    print("Finished clustering, proceeding with filtering.")