
import os
import math
import numpy as np
from collections import Counter

# my imports
from config import *


class DataPoint:
  """
  A simple class that handles a string example.
  """
  def __init__(self, string, index, only_string=True):
    """
    Params:
      :string:  String to be stored
      :index: Number of the line in the file from which this sentence was read
      :only_string: Whether to only store string
    """ 
    self.index=index
    self.string=string.strip("\n")
    self.cluster_index=0


class Cluster:
  """
  A class to handle one cluster in the clustering problem.
  """
  def __init__(self, medoid):
    """
    Params:
      :medoid:  Center of the cluster, data point object.
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

  def __init__(self, tag="full"):
    """
    Params:
      :tag: Can be either train, dev or test
    """
    self.tag=tag
    self.dist_matrix=np.ndarray(shape=(PROBLEM_HPARAMS["vocabulary_size"],
                                       PROBLEM_HPARAMS["vocabulary_size"]))
    self.treshold=DATA_FILTERING["treshold"]
    self.min_cluster_size=DATA_FILTERING["min_cluster_size"]
    self.clusters= {
      "Source" : [],
      "Target" : []
    }
    self.data_points= {
      "Source" : [],
      "Target" : []
    }

    # calculate number of clusters
    self.num_clusters = {
      "Source": 0,
      "Target": 0,
    }
    if self.tag=="train" or self.tag=="full":
      self.num_clusters["Source"]=DATA_FILTERING["source_clusters"]
      self.num_clusters["Target"]=DATA_FILTERING["target_clusters"]
    elif self.tag=="dev":
      self.num_clusters["Source"]=int(DATA_FILTERING["source_clusters"]
                                   * PROBLEM_HPARAMS["dataset_split"]["val"]
                                   / PROBLEM_HPARAMS["dataset_split"]["train"])
      self.num_clusters["Target"]=int(DATA_FILTERING["target_clusters"]
                                   * PROBLEM_HPARAMS["dataset_split"]["val"]
                                   / PROBLEM_HPARAMS["dataset_split"]["train"])
    else:
      self.num_clusters["Source"]=int(DATA_FILTERING["source_clusters"]
                                   * PROBLEM_HPARAMS["dataset_split"]["test"]
                                   / PROBLEM_HPARAMS["dataset_split"]["train"])
      self.num_clusters["Target"]=int(DATA_FILTERING["target_clusters"]
                                   * PROBLEM_HPARAMS["dataset_split"]["test"]
                                   / PROBLEM_HPARAMS["dataset_split"]["train"])

    self.output_data_dir=DATA_FILTERING["data_dir"]
    self.input_data_dir=FLAGS["data_dir"]
    self.type=DATA_FILTERING["filter_type"]

    # extra step to figure out in which split to put the results
    train_lines=self.count_lines("train")
    dev_lines=self.count_lines("dev")
    test_lines=self.count_lines("test")
    self.split_line_counts = {
      "train":  train_lines,
      "dev":    dev_lines,
      "test":   test_lines
    }

  # count the number of line in the given file
  def count_lines(self, file_tag="train"):
    file=open(os.path.join(self.input_data_dir, file_tag+"Source.txt"))
    num_lines=sum(1 for line in file)
    file.close()
    return num_lines

  # recursive function to create probability matrix
  def _create_tree(self, tr_word_matrix_tuple, tr_word_matrix, prob_matrix):
    # copy the lists because of recursion
    first_column=Counter(tr_word_matrix[0])

    # calculate probabilities for the current column
    for (row, index) in tr_word_matrix_tuple[0]:
      prob_matrix[index].append(first_column[row]/len(tr_word_matrix[0]))

    # check stopping criterion
    if len(tr_word_matrix)==1:
      return [[1]]
    else:
      for distinct_word in first_column:
        indices=[]
        # get the rows which we want to continue the tree
        for (word, index) in tr_word_matrix_tuple[0]:
          if word==distinct_word:
            indices.append(index)

        # transponate to be able to delete rows
        word_matrix=list(map(list, zip(*tr_word_matrix)))
        word_matrix_tuple=list(map(list, zip(*tr_word_matrix_tuple)))
        temp_mat=[]
        temp_mat_tuple=[]

        for row1, row2 in zip(word_matrix, word_matrix_tuple):
          if row2[0][1] in indices:
            temp_mat.append(row1)
            temp_mat_tuple.append(row2)

        # transponate back
        next_tr_matrix=list(map(list, zip(*temp_mat)))
        next_tr_matrix_tuple=list(map(list, zip(*temp_mat_tuple)))
        
        k=self._create_tree(next_tr_matrix_tuple[1:],
                            next_tr_matrix[1:],
                            prob_matrix)
      return prob_matrix

  # create input - target matrix pairs for distribution loss
  def create_bigram_matrix(self):
    self.read_inputs()
    self.clustering("Source")
    self.clustering("Target")

    # open data files
    fSource=open(os.path.join(self.output_data_dir, "DLOSS_source.txt"), "w")
    fTarget=open(os.path.join(self.output_data_dir, "DLOSS_target.txt"), "w")

    # loop through distinct inputs
    for cl in self.clusters["Source"]:
      max_len=0
      # loop through the targets to get the longest
      for target in cl.targets:
        sen_len=len(target.string.split())
        if DATA_FILTERING["max_length"]<sen_len:
          target.string=""
        elif max_len<sen_len:
          max_len=sen_len

      word_matrix=[]
      # loop through targets to pad them and create a word matrix
      for target in cl.targets:
        if target.string!="":
          words=target.string.split()
          word_matrix.append(words+["<pad>"]*(max_len-len(words)))

      tr_word_matrix=list(map(list, zip(*word_matrix)))
      prob_matrix=[[] for row in word_matrix]

      # add row indices to word matrix
      for i, target in enumerate(word_matrix):
        for j, word in enumerate(target):
          word_matrix[i][j]=(word, i)
      tr_word_matrix_tuple=list(map(list, zip(*word_matrix)))

      # recurse to create tree
      if word_matrix!=[]:
        fSource.write(cl.medoid.string+"\n")
        prob_matrix=self._create_tree(tr_word_matrix_tuple,
                                      tr_word_matrix,
                                      prob_matrix)

      # save target matrix to file
      for target_words, target_probs in zip(word_matrix, prob_matrix):
        for (word, index), prob in zip(target_words, target_probs):
          fTarget.write(word+":"+str(prob)+" ")
        fTarget.write("\n")
      fTarget.write("\n")

    fSource.close()
    fTarget.close()

  # main method that will run all the functions to do the filtering
  def run(self):
    if DATA_FILTERING["max_length"]>0:
      self.create_bigram_matrix()
    else:
      # if we have already done the clustering, don't redo it
      source_data=os.path.join(self.output_data_dir,
                               "..",
                               str(self.num_clusters["Source"])+"_clusters",
                               self.tag+"Source_cluster_elements.txt")
      target_data=os.path.join(self.output_data_dir,
                               "..",
                               str(self.num_clusters["Target"])+"_clusters",
                               self.tag+"Target_cluster_elements.txt")
      if os.path.isfile(source_data) and os.path.isfile(target_data):
        print("Cluster files are in "+self.output_data_dir+", filtering now.")
        self.load_clusters()
        self.filtering()

      else:
        print("No cluster files in "+self.output_data_dir+", clustering now.")
        self.read_inputs()
        self.clustering("Source")
        self.clustering("Target")
        self.save_clusters("Source")
        self.save_clusters("Target")
        self.filtering()

  # this function will read the data and make it ready for clustering
  def read_inputs(self):
    sources=open(os.path.join(self.input_data_dir, self.tag+"Source.txt"))
    targets=open(os.path.join(self.input_data_dir, self.tag+"Target.txt"))

    i=0
    for line in sources:
      self.data_points["Source"].append(self.DataPointClass(line, i, False))
      i+=1

    i=0
    for line in targets:
      self.data_points["Target"].append(self.DataPointClass(line, i, False))
      i+=1

    sources.close()
    targets.close()
    print("Finished reading "+self.tag+" data.")

  # load clusters from files
  def load_clusters(self):
    # open the data files
    source_clusters=open(
      os.path.join(self.output_data_dir,
                   "..",
                   str(self.num_clusters["Source"])+"_clusters",
                   self.tag+"Source_cluster_elements.txt"))
    target_clusters=open(
      os.path.join(self.output_data_dir,
                   "..",
                   str(self.num_clusters["Target"])+"_clusters",
                   self.tag+"Source_cluster_elements.txt"))

    # make a preloaded target cluster list
    self.clusters["Target"]=["" for i in range(self.num_clusters["Target"])]
    target_cluster_list=["" for i in range(self.split_line_counts["train"]
                                           +self.split_line_counts["dev"]
                                           +self.split_line_counts["test"])]
    # read the target clusters first
    for line in target_clusters:
      [index, line]=line.split(";")
      [source_medoid, pair, target_cluster]=line.strip("\n").split("<=====>")
      # list containing target medoid and target cluster index
      [target_medoid, target_cl_index]=target_cluster.split(":")
      target_cluster_list[int(index)]=[target_medoid, int(target_cl_index)]

    # load the clusters
    last_medoid="<-->"
    for line in source_clusters:
      [index, line]=line.split(";")
      [source_medoid, pair, target_cluster]=line.strip("\n").split("<=====>")
      [source, target]=pair.split("=")
      [target_medoid, target_cl_index]=target_cluster_list[int(index)]
      index=int(index)

      source_data_point=self.DataPointClass(source, index, True)
      target_data_point=self.DataPointClass(target, index, True)
      source_data_point.cluster_index=len(self.clusters["Source"])
      target_data_point.cluster_index=target_cl_index

      # check if this is a new medoid (source side)
      if last_medoid!=source_medoid:
        # add medoid to cluster
        dp=self.DataPointClass(source_medoid, index=0, only_string=True)
        self.clusters["Source"].append(self.ClusterClass(dp))
      self.clusters["Source"][-1].add_element(source_data_point)
      self.clusters["Source"][-1].targets.append(target_data_point)

      # target side
      if self.clusters["Target"][target_cl_index]=="":
        dp=self.DataPointClass(target_medoid, index=0, only_string=True)
        self.clusters["Target"][target_cl_index]=self.ClusterClass(dp)
      self.clusters["Target"][target_cl_index].add_element(target_data_point)
      self.clusters["Target"][target_cl_index].targets.append(source_data_point)

      last_medoid=source_medoid

    source_clusters.close()
    target_clusters.close()

  # find the point that minimizes mean distance within a cluster
  def find_medoid(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """
    for cluster in self.clusters[data_tag]:
      print("Finding medoids.")
      big_sum=0
      for element1 in cluster.elements:
        small_sum=0
        for element2 in cluster.elements:
          small_sum+=element1.similarity(element2, self.dist_matrix)

        if small_sum>big_sum:
          big_sum=small_sum
          cluster.medoid = element1

      # clear elements after we finished with one cluster
      cluster.elements.clear()
      cluster.targets.clear()

  # find nearest medoid for a data point
  def find_nearest_medoid(self, data_point):
    NotImplementedError

  # for each data_point find a cluster
  def cluster_points(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """
    # reverse data tag
    rev_tag="Target" if data_tag=="Source" else "Source"
    
    for i, data_point in enumerate(self.data_points[data_tag]):
      #if i%1000==0:
      print(str(i))
      nearest_medoid=self.find_nearest_medoid(data_point, data_tag)
      self.clusters[data_tag][nearest_medoid].add_element(data_point)

      self.clusters[data_tag][nearest_medoid].targets.append(
        self.data_points[rev_tag][i])

      data_point.cluster_index=nearest_medoid

  def stop_clustering(self,
                      data_tag,
                      cluster_names,
                      cluster_names_old,
                      count,
                      counts):
    """
    Params:
      :data_tag: Whether it's source or target data
      :cluster_names: String of medoids from previous iteration
      :cluster_names_old: String of medoids from 2 iterations ago
      :count: Number of clustering loops so far
    """ 
    count_difference=0
    count_difference_old=0
    for i in range(self.num_clusters[data_tag]):
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

    # check if no. of medoids changed is the same for the last 6 iterations
    same_counts=True
    counts.append(count_difference)
    if len(counts)>6:
      counts=list(counts[1:])
    for i, c in enumerate(counts[:-1]):
      if c!=counts[i+1]:
        same_counts=False

    # exit if there is no change or we are stuck in a loop
    exit=False
    if count_difference==0 or count_difference_old==0 or same_counts:
      exit=True
    return exit, cluster_names, cluster_names_old, counts

  # do the clustering of sources and targets
  def clustering(self, data_tag):
    return NotImplementedError

  # return a list of indices, showing which clusters should be filtered out
  def get_filtered_indices(self, source):
    """
    Params:
      :source: the cluster that we want to filter (either Source or Target)
    """
    indices=[]
    for num_cl, cluster in enumerate(self.clusters[source]):
      # error guarding for the case when loading clusters
      if cluster!="":
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
        cluster.entropy=-entropy

        # filter
        if cluster.entropy>self.treshold:
          indices.append(num_cl)
          print('Medoid: "'+cluster.medoid.string+'" got filtered.')

    print("Finished filtering "+source+" data.")
    return indices
    
  # do the filtering of the dataset
  def filtering(self):
    # these are not needed anymore
    self.data_points["Source"].clear()
    self.data_points["Target"].clear()

    source_indices=self.get_filtered_indices("Source")
    target_indices=self.get_filtered_indices("Target")

    # open files
    file_dict={}
    # we have to open 6 files in this case
    if self.tag=="full":
      name_list = ["trainS", "trainT", "devS", "devT", "testS", "testT"]
      file_list = list(self.open_6_files())
      file_dict = dict(zip(name_list, file_list))
    
    # handle all cases and open files
    if self.type=="target_based" or self.type=="both":
      file_dict["source_entropy"]=open(
        os.path.join(self.output_data_dir,
                     self.tag+"Source_cluster_entropies.txt"), "w")
    if self.type=="source_based" or self.type=="both":
      file_dict["target_entropy"]=open(
        os.path.join(self.output_data_dir,
                     self.tag+"Target_cluster_entropies.txt"), "w")
    file_dict[self.tag+"source_file"]=open(
      os.path.join(self.output_data_dir, self.tag+"Source.txt"), "w")
    file_dict[self.tag+"target_file"]=open(
      os.path.join(self.output_data_dir, self.tag+"Target.txt"), "w")

    # save data
    self.save_filtered_data(source_indices, target_indices, file_dict)
    self.close_n_files(file_dict)

  # save the new filtered datasets
  def save_filtered_data(self,
                         source_indices=[],
                         target_indices=[],
                         file_dict={}):
    """
    Params:
      :source_indices: Indices of source clusters that will be filtered.
      :target_indices: Indices of target clusters that will be filtered.
      :file_dict: Dictionary containing all the files that we want to write.
    """
    # function for writing the dataset to file
    def save_dataset(source):
      for num_cl, cluster in enumerate(self.clusters[source]):
        # filter errors due to cluster loading
        if cluster!="":
          # write cluster entropies
          file_dict[source.lower()+"_entropy"].write(
            cluster.medoid.string+";"
            +str(cluster.entropy)+";"
            +str(len(cluster.elements))+"\n")

          indices=source_indices if source=="Source" else target_indices
          clusters_too_small=len(cluster.elements)<self.min_cluster_size
          # make sure that in "both" case only run this once
          if (source=="Source" or self.type!="both") \
              and (num_cl not in indices or clusters_too_small):
            # filter one side
            for num_el, element in enumerate(cluster.elements):
              target_cl=cluster.targets[num_el].cluster_index
              clusters_too_small=len(
                self.clusters["Target"][target_cl].elements) \
                < self.min_cluster_size
              # check both sides in "both" case
              if ((target_cl not in target_indices or clusters_too_small)
                  or self.type!="both"):
                source_string=element.string+"\n"
                target_string=cluster.targets[num_el].string+"\n"

                # reverse if Target
                if source=="Target":
                  tmp=source_string
                  source_string=target_string
                  target_string=tmp
                file_dict[self.tag+"source_file"].write(source_string)
                file_dict[self.tag+"target_file"].write(target_string)

                # write to separate files if we do split after clustering
                if self.tag=="full":
                  if element.index<self.split_line_counts["train"]:
                    file_dict["trainS"].write(source_string)
                    file_dict["trainT"].write(target_string)
                  elif element.index<(self.split_line_counts["train"]
                                      +self.split_line_counts["dev"]):
                    file_dict["devS"].write(source_string)
                    file_dict["devT"].write(target_string)
                  else:
                    file_dict["testS"].write(source_string)
                    file_dict["testT"].write(target_string)

    # write source entropies to file
    if self.type=="target_based" or self.type=="both":
      save_dataset("Source")
    # write target entropies to file
    if self.type=="source_based" or self.type=="both":
      save_dataset("Target")

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
          str(data_point.index)+";"
          +cluster.medoid.string+"<=====>"
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

  # open the 6 files
  def open_6_files(self):
    trainS = open(os.path.join(self.output_data_dir, 'trainSource.txt'), 'w')
    trainT = open(os.path.join(self.output_data_dir, 'trainTarget.txt'), 'w')
    devS = open(os.path.join(self.output_data_dir, 'devSource.txt'), 'w')
    devT = open(os.path.join(self.output_data_dir, 'devTarget.txt'), 'w')
    testS = open(os.path.join(self.output_data_dir, 'testSource.txt'), 'w')
    testT = open(os.path.join(self.output_data_dir, 'testTarget.txt'), 'w')

    return trainS, trainT, devS, devT, testS, testT

  # close the 6 files to write the processed data into
  def close_n_files(self, files):
    for file_name in files:
      files[file_name].close()