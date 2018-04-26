import numpy as np
import os
from tensorflow.python import pywrap_tensorflow

# my imports
from config import *
from data_filtering.filter_problem import FilterProblem

# a class to handle each data point
class DataPoint:
  """
  A class that handles a hash example.
  """
  def __init__(self, string, only_string=True):
    """
    Params:
      :string:  String to be stored
      :vocab_dict: Vocabulary dictionary with embedding weights.
      :only_string: Whether to only store string
    """ 
    self.string=string.strip("\n")
    self.ids=[]

  # convert string to vocab ids
  def convert_to_ids(self, vocab_dict):
    """
    Params:
      :vocab_dict: Vocabulary dictionary that contains embedding weights
    """
    words=self.string.split()

    for word in words:
      if word in vocab_dict:
        self.ids.append(vocab_dict[word])
      else:
        self.ids.append(vocab_dict["<unk>"])

  # compute the distance of two embeddings
  def emb_dist(id1, id2):
    return np.linalg.norm(id1-id2)

  # distance metric between this and another sentence
  def distance(self, other):
    # TODO: what is the range of this distance?

    self_ids=list(self.ids)
    self_words=self.string.split()
    other_ids=list(other.ids)
    other_words=other.string.split()

    indices=[]
    # first delete the ones that are the same
    for i, (word, weight) in enumerate(zip(self_words, self_ids)):
      if word in other_words:
        self_ids.remove(weight)
        other_ids.remove(weight)

    distances=[[] for x in range(len(self_ids))]
    # compute distance in one way
    total=0
    for i, self_id in enumerate(self_ids):
      minimum=1

      for other_id in other_ids:
        distances[i].append(self.emb_dist(self_id, other_id))
        # compare
        if distances[i][-1]<minimum:
          minimum=distances[i][-1]
      total+=minimum
    first_sum=total/len(self_ids)

    # compute distance in the other way
    total=0
    for j in range(len(other_ids)):
      minimum=1

      for i in range(len(self_ids)):
        if distances[i][j]<minimum:
          minimum=distance[i][j]
      total+=minimum
    second_sum=total/len(other_ids)

    return (first_sum+second_sum)/2

  # computes a similarity metric between two sentences
  def similarity(self, other):
    return 1-self.distance(self, other)

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
      dist=data_point.distance(medoid)

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
                          self.num_clusters)
    for i in range(self.num_clusters):
      cl=self.ClusterClass(self.data_points[data_tag][medoids[i]])
      self.clusters[data_tag].append(cl)

    # for each data_point find a cluster
    self.cluster_points(data_tag)

    # these will be needed for the stopping criterion
    cluster_names=[self.clusters[data_tag][i].medoid.string
                    for i in range(self.num_clusters)]
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
    # make the weight dir if it doesn't exist
    if not os.path.exists(os.path.join(FLAGS["train_dir"], "weights")):
      os.makedirs(os.path.join(FLAGS["train_dir"], "weights"))

    reader = pywrap_tensorflow.NewCheckpointReader(self.ckpt_file_name)
    lines=reader.debug_string().decode("utf-8").split("\n")

    # save the individual weight tensors
    weight_names=[]
    for line in lines:
      if "symbol_modality" in line and "training" not in line:
        weight=line.split()[0]
        weight_no=weight.split("/")[-1]

        output=open(
          os.path.join(FLAGS["train_dir"], "weights", str(weight_no)+".txt"))
        array=reader.get_tensor(tensor_name)
        w, h=array.shape
        for i in range(w):
          for j in range(h):
            output.write(str(array[i, j])+";")
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

    word_idx=0
    # read through the weight files
    for i in range(self.num_shards):
      weight_file=open(FLAGS["train_dir"], "weights", "weights_"+str(i)+".txt")

      for embedding in weight_file:
        params=embedding.split(";")[:-1]
        self.vocab_dict[vocab_list[word_idx]]=[]

        # save to the vocab dict
        for j, param in enumerate(params):
          self.vocab_dict[vocab_list[word_idx]].append(float(param))
        word_idx+=1

      weight_file.close()  