import numpy as np
import os
import random
from tensorflow.python import pywrap_tensorflow
from collections import Counter
from multiprocessing import Process
import scipy

# my imports
from config import *
from data_filtering.filter_problem import FilterProblem

# temporary helper function to load a vocabulary
def _load_vocab():
  vocab=open(
    os.path.join(FLAGS["data_dir"],
                 "vocab.chatbot."+str(PROBLEM_HPARAMS["vocabulary_size"])))
  vocab_dict={}
  # read the vocab file
  i=0
  for word in vocab:
    vocab_dict[word.strip("\n")]=i
    i+=1
  vocab.close()

  return vocab_dict

# a class to handle each data point
class DataPoint:
  """
  A class that handles a hash example.
  """
  vocab_dict=_load_vocab()
  def __init__(self, string, index, only_string=True):
    """
    Params:
      :string:  String to be stored
      :index: Number of the line in the file from which this sentence was read
      :only_string: Whether to only store string
    """
    self.index=index
    self.string=string.strip("\n")
    self.words=self.string.split()

    # replace out of vocabulary words
    for i, word in enumerate(self.words):
      if word not in DataPoint.vocab_dict:
        self.words[i]="<unk>"
      self.words[i]=DataPoint.vocab_dict[self.words[i]]

    # transform to counter
    self.words=Counter(self.words)

  # distance metric between this and another sentence
  def distance(self_counter, other_counter, dist_matrix):
    """
    Params:
      :other: The other sentence to which we calculate distance
      :dist_matrix: Distance matrix for all words in vocab
    """
    def word_sum(self_counter, other_counter):
      # compute distance in one way
      dist_sum=0
      for self_word in self_counter:
        minimum=1
        for other_word in other_counter:
          dist=dist_matrix[self_word, other_word]
          if dist<minimum:
            minimum=dist

        count=self_counter[self_word]
        dist_sum+=count*minimum

      # normalize  
      self_length=len(self_counter)
      if self_length!=0:
        dist_sum=dist_sum/self_length
      return dist_sum

    first_sum=word_sum(self_counter.words, other_counter.words)
    second_sum=word_sum(other_counter.words, self_counter.words)
    return (first_sum+second_sum)/2

  # computes a similarity metric between two sentences
  def similarity(self, other, dist_matrix):
    return -self.distance(other, dist_matrix)

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

  @property
  def weights_folder(self):
    return FLAGS["train_dir"]+"/weights"+str(DATA_FILTERING["ckpt_number"])

  # find nearest medoid for a data point
  def find_nearest_medoid(self, data_point, data_tag=""):
    """
    Params:
      :data_point: Data point for which we want to find the nearest medoid
      :data_tag: Whether it's source or target data
    """
    min_distance=0
    for i, cluster in enumerate(self.clusters[data_tag]):
      dist=data_point.distance(cluster.medoid, self.dist_matrix)

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
    if not os.path.exists(self.weights_folder):
      os.makedirs(self.weights_folder)
    # first we have to extract the weights from the model
    if len(os.listdir(self.weights_folder)) !=self.num_shards+1:
      self.extract_weights()
      self.create_vocab_matrix()

    # load vocab distance matrix
    if data_tag=="Source":
      self.load_distance_matrix()
      
    # create a sentence distance matrix
    processes=[]
    for i in range(16):
      p=Process(target=self.create_sentence_distance_matrix, args=(data_tag,i))
      processes.append(p)
      p.start()

    # stop processes
    for process in processes:
      process.join()

    """
    # initialize clusters
    self.load_sentence_matrix(data_tag)
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
    """

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
          os.path.join(self.weights_folder, str(weight_no)+".txt"), "w")
        array=reader.get_tensor(weight)
        w, h=array.shape
        for i in range(w):
          for j in range(h):
            output.write(str(array[i, j])+";")
          output.write("\n")
        output.close()

  # create vocab distance matrix
  def create_vocab_matrix(self):
    # load the vocab
    vocab=open(
      os.path.join(self.input_data_dir,
                   "vocab.chatbot."+str(PROBLEM_HPARAMS["vocabulary_size"])))
    vocab_dict={}
    vocab_list=[]
    # read the vocab file
    for word in vocab:
      vocab_list.append(word.strip("\n"))
    vocab.close()

    word_idx=0
    # read through the weight files
    for i in range(self.num_shards):
      weight_file=open(
        os.path.join(self.weights_folder, "weights_"+str(i)+".txt"))

      for embedding in weight_file:
        params=embedding.split(";")[:-1]
        vocab_dict[vocab_list[word_idx]]=[]

        # save to the vocab dict
        for j, param in enumerate(params):
          vocab_dict[vocab_list[word_idx]].append(float(param))
        
        weight=vocab_dict[vocab_list[word_idx]]
        vocab_dict[vocab_list[word_idx]]=np.array(weight)
        word_idx+=1
      weight_file.close()

    matrix_file=open(
      os.path.join(self.weights_folder, "distance_matrix.txt"), "w")
    matrix_file.write(" ;")
    for key in vocab_dict:
      matrix_file.write(key+";")
    # create the weight matrix and save it to file
    i=0
    for key1 in vocab_dict:
      i+=1
      print(i)
      matrix_file.write("\n"+key1+";")
      for key2 in vocab_dict:
        dist=np.linalg.norm(np.subtract(vocab_dict[key1], vocab_dict[key2]))
        matrix_file.write(str(dist)+";")
    matrix_file.close()

  # load distance matrix for vocab weights
  def load_distance_matrix(self):
    matrix_file=open(
      os.path.join(self.weights_folder, "distance_matrix.txt"))

    i=-1
    vocab_list=[]
    # load the distances into a dictionary
    for line in matrix_file:
      if i==-1:
        vocab_list=line.split(";")[1:-1]
      else:
        distances=line.split(";")[:-1]
        for j, dist in enumerate(distances[1:]):
          self.dist_matrix[i, j]=float(dist)
      i+=1

    matrix_file.close()

  # create a sentence distance matrix
  def create_sentence_distance_matrix(self, data_tag, pid):
    """
    Params:
      :pid: process id of this function (used to split data)
      :dist_matrix: vocab distance matrix
    """
    # save the created distance matrix
    out=open(
      os.path.join(
        self.input_data_dir, data_tag+"SentenceMatrix"+str(pid)+".txt"), "w")

    length=len(self.data_points[data_tag])
    for dp1 in self.data_points[data_tag][int(pid/16*length):int((pid+1)/16*length)]:
      for dp2 in self.data_points[data_tag]:
        out.write(str(dp1.distance(dp2, self.dist_matrix))+";")
      out.write("\n")
    out.close()

  # load the sentence matrix
  def load_sentence_matrix(self, data_tag):
    self.sentence_matrix={}
    matrix_file=open(
      os.path.join(self.input_data_dir, data_tag+"SentenceMatrix.txt"))

    i=0
    for line in matrix_file:
      distances=line.split(";")[:-1]
      for j, dist in enumerate(distances):
        self.sentence_matrix[(i, j)]=float(dist)
      i+=1

    matrix_file.close()