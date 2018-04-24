import os
import math
from collections import Counter

# my imports
from config import *



class DataPoint:
  """
  A simple class that handles a string example.
  """
  def __init__(self, string, only_string=True):
    """
    Params:
      :string:  String to be stored
      :only_string: Whether to only store string
    """ 
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
    self.entropy=0

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
      "Source" : [],
      "Target" : []
    }
    self.data_points= {
      "Source" : [],
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
    # if we have already done the clustering, don't redo it
    if os.path.isfile(os.path.join(self.output_data_dir, 
                                   self.tag+"Source_cluster_elements.txt")) \
      and os.path.isfile(os.path.join(self.output_data_dir, 
                                      self.tag+"Target_cluster_elements.txt")):

      print("Found cluster files in "+self.output_data_dir+", filtering next.")
      self.load_clusters()
      self.filtering()

    else:
      print("No cluster files in "+self.output_data_dir+", clustering now.")
      self.read_inputs()
      self.clustering("Source")
      self.clustering("Target")
      self.filtering()

  # this function will read the data and make it ready for clustering
  def read_inputs(self):
    sources=open(os.path.join(self.input_data_dir, self.tag+"Source.txt"))
    targets=open(os.path.join(self.input_data_dir, self.tag+"Target.txt"))

    for line in sources:
      self.data_points["Source"].append(self.DataPointClass(line))

    for line in targets:
      self.data_points["Target"].append(self.DataPointClass(line))

    sources.close()
    targets.close()
    print("Finished reading "+self.tag+" data.")

  # load clusters from files
  def load_clusters(self):
    clusters=open(os.path.join(self.output_data_dir, 
                               self.tag+"Source_cluster_elements.txt"))

    # make a preloaded target cluster list
    self.clusters["Target"]=["" for i in range(self.num_clusters)]

    # load the clusters
    last_medoid="<-->"
    for line in clusters:
      [source_medoid, pair, target_cluster]=line.strip("\n").split("<=====>")
      [source, target]=pair.split("=")
      [target_medoid, target_cl_index]=target_medoid.split(":")

      source_data_point=self.DataPointClass(source, only_string=True)
      target_data_point=self.DataPointClass(target, only_string=True)
      source_data_point.cluster_index=len(self.clusters["Source"])
      target_data_point.cluster_index=int(target_cl_index)

      # check if this is a new medoid (source side)
      if last_medoid!=source_medoid:
        # add medoid to cluster
        cl=self.ClusterClass(self.DataPointClass(medoid, only_string=True))
        self.clusters["Source"].append(cl)
      self.clusters["Source"][-1].add_element(source_data_point)
      self.clusters["Source"][-1].targets.append(target_data_point)

      # target side
      if self.clusters["Target"][target_cl_index]=="":
        dp=self.DataPointClass(target_medoid, only_string=True)
        self.clusters["Target"][target_cl_index]=self.ClusterClass(dp)
      self.clusters["Target"][target_cl_index].add_element(target_data_point)
      self.clusters["Target"][target_cl_index].targets.append(source_data_point)

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
          small_sum+=self.clusters[data_tag][i].elements[j1].distance(
                      self.clusters[data_tag][i].elements[j2])

        if small_sum>big_sum:
          big_sum=small_sum
          self.clusters[data_tag][i].medoid = (
            self.clusters[data_tag][i].elements[j1])

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
    for i, data_point in enumerate(self.data_points[data_tag]):
      nearest_medoid=self.find_nearest_medoid(data_point)
      self.clusters[data_tag][nearest_medoid].add_element(data_point)

      # reverse data tag
      rev_tag="Target" if data_tag=="Source" else "Source"
      self.clusters[data_tag][nearest_medoid].targets.append(
        self.data_points[rev_tag][i])

      data_point.cluster_index=nearest_medoid

  def stop_clustering(self, data_tag, cluster_names, cluster_names_old, count):
    """
    Params:
      :data_tag: Whether it's source or target data
      :cluster_names: String of medoids from previous iteration
      :cluster_names_old: String of medoids from 2 iterations ago
      :count: Number of clustering loops so far
    """ 
    count_difference=0
    count_difference_old=0
    for i in range(self.num_clusters):
      # check strings from previous iteration to see if they are the same
      if self.clusters[data_tag][i].medoid.string!=cluster_names[i]:
        count_difference+=1
        print(cluster_names[i]+"--->"+self.clusters[data_tag][i].medoid.string)
        cluster_names[i]=self.clusters[data_tag][i].medoid.string

      # check strings from two loops ago, to see if they are the same
      if self.clusters[data_tag][i].medoid.string!=cluster_names_old[i]:
        count_difference_old+=1
        if count %2==0:
          cluster_names_old[i]=self.clusters[data_tag][i].medoid.string
    print("==================================================")
    print("==================================================")

    # exit if there is no change or we are stuck in a loop
    exit=True if count_difference==0 or count_difference_old==0 else False
    return exit, cluster_names, cluster_names_old

  # do the clustering of sources and targets
  def clustering(self, data_tag):
    return NotImplementedError

  # return a list of indices, showing which clusters should be filtered out
  def get_filtered_indices(self, source):
    """
    Params:
      :source: the cluster that we want to filter (either Source or Target)
    """
    target_string="Target" if source=="Source" else "Source"
    entropy_stats=open(
      os.path.join(self.output_data_dir,
                   self.tag+source+"_cluster_entropies.txt"), "w")
    source_file=open(os.path.join(self.output_data_dir,
                                  self.tag+source+".txt"), "w")
    target_file=open(os.path.join(self.output_data_dir,
                                  self.tag+target_string+".txt"), "w")

    indices=[]
    for num_cl, cluster in enumerate(self.clusters[source]):
      # build a distribution for the current cluster, based on the targets
      distribution=Counter()
      for target in cluster.targets: 
        if target.cluster_index in distribution:
          distribution[target.cluster_index]+=1
        else:
          distribution[target.cluster_index]=1

      # calculate entropy
      entropy=0
      for cl_index in distribution:
        if len(cluster.targets)>1:
          probability=distribution[cl_index]/len(cluster.targets)
          entropy+=probability*math.log(probability, 2)

      # normalize entropy between 0 and 1, and save it
      if len(cluster.targets)>1:
        entropy=-entropy/math.log(len(cluster.targets), 2)
      cluster.entropy=entropy

      # save the filtering results
      if self.type!="both":
        self.save_filtered_data(cluster,
                                entropy_stats,
                                source_file, 
                                target_file)
      # filter
      if entropy>self.treshold:
        indices.append(num_cl)
        print('Medoid: "'+cluster.medoid.string+'" got filtered.')

    print("Finished filtering.")
    source_file.close()
    target_file.close()
    entropy_stats.close()

    return indices
    
  # do the filtering of the dataset
  def filtering(self):
    # these are not needed anymore
    self.data_points["Source"].clear()
    self.data_points["Target"].clear()

    if self.type=="target_based" or self.type=="both":
      source_indices=self.get_filtered_indices("Source")
    if self.type=="source_based" or self.type=="both":
      target_indices=self.get_filtered_indices("Target")

    # this is a special case, has to be handled separately
    if self.type=="both":
      source_entropy=open(
        os.path.join(self.output_data_dir,
                     self.tag+"Source_cluster_entropies.txt"), "w")
      target_entropy=open(
        os.path.join(self.output_data_dir,
                     self.tag+"Target_cluster_entropies.txt"), "w")
      source_file=open(os.path.join(self.output_data_dir,
                                    self.tag+source+".txt"), "w")
      target_file=open(os.path.join(self.output_data_dir,
                                    self.tag+target_string+".txt"), "w")

      # write entropies to file
      for cluster in self.clusters["Target"]:
        target_entropy.write(
          cluster.medoid.string+";"+str(cluster.entropy)+"\n")

      for num_cl, cluster in enumerate(self.clusters["Source"]):
        source_entropy.write(
          cluster.medoid.string+";"+str(cluster.entropy)+"\n")

        # double filtering (source and target)
        if num_cl not in source_indices:
          for num_el, element in enumerate(cluster.elements):
            if cluster.targets[num_el].cluster_index not in target_indices:
              source_file.write(element.string+"\n")
              target_file.write(cluster.targets[num_el].string+"\n")

      source_file.close()
      target_file.close()
      source_entropy.close()
      target_entropy.close()

  # save the filtering results
  def save_filtered_data(self, cluster, entropy_stats, source, target):
    """
    Params:
      :cluster: Cluster to be saved
      :entropy_stats: Cluster entropies are saved to this file
      :source: Filtered source sentences are saved here
      :target: Filtered target sentences are saved here
    """
    entropy_stats.write(cluster.medoid.string+";"+str(cluster.entropy)+"\n")

    if cluster.entropy<=self.treshold:
      for num_el, element in enumerate(cluster.elements):
        source.write(element.string+"\n")
        target.write(cluster.targets[num_el].string+"\n")

  # save clusters and their elements to files
  def save_clusters(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """
    output=open(os.path.join(self.output_data_dir, 
                             self.tag+data_tag+"_cluster_elements.txt"), "w")
    medoid_counts=[]
    rev_tag="Target" if data_tag=="Source" else "Source"
    for cluster in self.clusters[data_tag]:
      medoid_counts.append((cluster.medoid.string, len(cluster.elements)))

      # save together the source and target medoids and elements
      for data_point, target in zip(cluster.elements, cluster.targets):
        output.write(
          cluster.medoid.string+"<=====>"
          +data_point.string+"="
          +target.string+"<=====>"
          +self.clusters[rev_tag][target.cluster_index].medoid.string+":"
          +str(target.cluster_index)+"\n")
    output.close()

    # save the medoids and the count of their elements, in decreasing order
    output=open(os.path.join(self.output_data_dir, 
                             self.tag+data_tag+"_clusters.txt"), "w")
    medoids=sorted(medoid_counts, key=lambda count: count[1], reverse=True)

    for medoid in medoids:
      output.write(medoid[0]+":"+str(medoid[1])+"\n")
    output.close()

    print("Finished clustering, proceeding with filtering.")