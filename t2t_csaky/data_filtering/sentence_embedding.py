import numpy as np
import os
import random
from tensorflow.python import pywrap_tensorflow
from collections import Counter

# my imports
from config import *
from data_filtering.filter_problem import FilterProblem

# a class to handle each data point
class DataPoint:
  """
  A class that handles a hash example.
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
    self.ids=[]

  # convert string to vocab ids
  def convert_to_ids(self, vocab_dict):
    """
    Params:
      :vocab_dict: Vocabulary dictionary that contains embedding weights
    """
    self.words=self.string.split()

    for word in self.words:
      if word in vocab_dict:
        self.ids.append(vocab_dict[word])
      else:
        self.ids.append(vocab_dict["<unk>"])

  # compute the distance of two embeddings
  def emb_dist(self, id1, id2):
    return np.linalg.norm(np.subtract(np.array(id1), np.array(id2)))

  # distance metric between this and another sentence
  def distance(self, other):
    # TODO: redo this based on what recski told me

    self_word_id=dict(zip(self.words, self.ids))
    other_word_id=dict(zip(other.words, other.ids))
    self_counter=Counter(self.words)
    other_counter=Counter(other.words)

    self_minus_other=self_counter-other_counter
    other_minus_self=other_counter-self_counter
    distances={}

    # compute distance in one way
    first_sum=0
    for self_word in self_minus_other:
      minimum=1
      for other_word in other_minus_self:
        dist=self.emb_dist(self_word_id[self_word], other_word_id[other_word])
        distances[(self_word, other_word)]=dist
        if dist<minimum:
          minimum=dist

      count=self_minus_other[self_word]
      first_sum+=count*minimum
    # normalize  
    self_length=len(self_minus_other)
    if self_length!=0:
      first_sum=first_sum/self_length

    # compute distance in the other way
    second_sum=0
    for other_word in other_minus_self:
      minimum=1
      for self_word in self_minus_other:
        dist=distances[(self_word, other_word)]
        if dist<minimum:
          minimum=dist

      count=other_minus_self[other_word]
      second_sum+=count*minimum
    # normalize
    other_length=len(other_minus_self)
    if other_length!=0:
      second_sum=second_sum/other_length

    # if one sentence contains the other
    if self_length==0 or other_length==0:
      return abs(self_length-other_length) / max(len(self.ids), len(other.ids))
    return (first_sum+second_sum)/2

  # computes a similarity metric between two sentences
  def similarity(self, other):
    return 1-self.distance(other)

class SentenceEmbedding(FilterProblem):
  """
  A class that does clustering based on hashes from the datasketch library.
  """
  @property
  def DataPointClass(self):
    return DataPoint

  @property
  def ckpt_file_name(self):
    return FLAGS["train_dir"]+"/model.ckpt-"+str(DATA_FILTERING["ckpt_number"])

  # find nearest medoid for a data point
  def find_nearest_medoid(self, data_point, data_tag=""):
    """
    Params:
      :data_point: Data point for which we want to find the nearest medoid
      :data_tag: Whether it's source or target data
    """
    min_distance=0
    for i, medoid in enumerate(self.clusters[data_tag]):
      dist=data_point.distance(medoid.medoid)

      if i==0 or dist<min_distance:
        min_distance=dist
        nearest_medoid=i
    return nearest_medoid

  # do the clustering of sources and targets
  def clustering(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data
    """
    # which model
    self.num_shards=SEQ2SEQ_HPARAMS["embed_num_shards"]
    if FLAGS["model"]=="transformer":
      self.num_shards=TRANSFORMER_HPARAMS["embed_num_shards"]

    # make the weight dir if it doesn't exist
    if not os.path.exists(os.path.join(FLAGS["train_dir"], "weights")):
      os.makedirs(os.path.join(FLAGS["train_dir"], "weights"))
    # first we have to extract the weights from the model
    if len(os.listdir(os.path.join(FLAGS["train_dir"], "weights"))) \
        !=self.num_shards:
      self.extract_weights()

    # then we have to load them to one big matrix
    if data_tag=="Source":
      self.load_weights()

    # upgrade each data point with a weight list
    for dp in self.data_points[data_tag]:
      dp.convert_to_ids(self.vocab_dict)

    # initialize clusters
    medoids=random.sample(range(len(self.data_points[data_tag])),
                          self.num_clusters[data_tag])
    for i in range(self.num_clusters[data_tag]):
      cl=self.ClusterClass(self.data_points[data_tag][medoids[i]])
      self.clusters[data_tag].append(cl)

    # for each data_point find a cluster
    self.cluster_points(data_tag)

    # these will be needed for the stopping criterion
    cluster_names=[self.clusters[data_tag][i].medoid.string
                    for i in range(self.num_clusters[data_tag])]
    cluster_names_old=list(cluster_names)
    count=0
    counts=[]
    exit=False

    # clustering loop
    while not exit:
      count+=1

      # find the point that minimizes the mean distance within a cluster
      self.find_medoid(data_tag)

      # assign each point to the new medoids
      self.cluster_points(data_tag)

      # check stopping criterions
      exit, cluster_names, cluster_names_old, counts = \
        self.stop_clustering(data_tag,
                             cluster_names,
                             cluster_names_old,
                             count,
                             counts)

  # extract embedding weights
  def extract_weights(self):
    reader = pywrap_tensorflow.NewCheckpointReader(self.ckpt_file_name)
    lines = reader.debug_string().decode("utf-8").split("\n")

    # save the individual weight tensors
    weight_names=[]
    for line in lines:
      if "symbol_modality" in line and "training" not in line:
        weight=line.split()[0]
        weight_no=weight.split("/")[-1]

        output=open(
          os.path.join(FLAGS["train_dir"],
                   "weights",
                   str(weight_no)+".txt"), "w")
        array=reader.get_tensor(weight)
        w, h=array.shape
        for i in range(w):
          for j in range(h):
            output.write(str(array[i, j])+";")
          output.write("\n")
        output.close()

  # process embedding weights      
  def load_weights(self):
    vocab=open(
      os.path.join(self.input_data_dir,
                   "vocab.chatbot."+str(PROBLEM_HPARAMS["vocabulary_size"])))

    self.vocab_dict={}
    vocab_list=[]
    # read the vocab file
    for word in vocab:
      vocab_list.append(word.strip("\n"))
    vocab.close()

    word_idx=0
    # read through the weight files
    for i in range(self.num_shards):
      weight_file=open(
        os.path.join(FLAGS["train_dir"], "weights", "weights_"+str(i)+".txt"))

      for embedding in weight_file:
        params=embedding.split(";")[:-1]
        self.vocab_dict[vocab_list[word_idx]]=[]

        # save to the vocab dict
        for j, param in enumerate(params):
          self.vocab_dict[vocab_list[word_idx]].append(float(param))
        
        word_idx+=1
      weight_file.close()