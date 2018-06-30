
import os
import numpy as np

_use_faiss = True

try:

  import faiss

except ImportError:
  print('Failed to import faiss, using SKLearn clustering instead.')
  from sklearn.cluster import KMeans
  _use_faiss = False

# my imports
from . import filter_problem
from config import *


class DataPoint(filter_problem.DataPoint):
  """
  A simple class that handles a string example.
  """
  def __init__(self, meaning_vector, string, index, only_string=True):
    super().__init__(string, index, only_string)
    self.meaning_vector = meaning_vector


class RNNState(filter_problem.FilterProblem):

  @property
  def DataPointClass(self):
    return DataPoint

  def clustering(self, data_tag):
    sentence_mv_path = self._data_path('{}States'.format(data_tag), 'npy')
    meaning_vectors = np.load(sentence_mv_path)
    k = 3
    niter = 10

    centroids, kmeans = calculate_centroids(meaning_vectors, k, niter)
    clusters = [filter_problem.Cluster(centroid) for centroid in centroids]

    for data_point in self.data_points:
      cluster_index = calculate_nearest_index(
        data_point.meaning_vector, kmeans)

      clusters[cluster_index].add_element(data_point)
      data_point.cluster_index = cluster_index

  # this function will read the data and make it ready for clustering
  def read_inputs(self):
    def read_sentences(file):
      sentences = []
      with open(file, 'r', encoding='utf-8') as f:
        for line in f:
          sentences.append(line.strip('\n'))
      return sentences

    def read(data_tag):
      # if the encodings exists they will not be generated again
      # TODO meaning vector and decode output path fix
      regular_data_path = self._data_path(data_tag, 'txt')
      ordered_data_path = self._data_path('{}Ordered'.format(data_tag), 'txt')
      # sentence meaning vector
      sentence_mv_path = self._data_path('{}States'.format(data_tag), 'npy')

      if (not os.path.exists(ordered_data_path) or
          not os.path.exists(sentence_mv_path)):
        generate_encoder_states()

      meaning_vectors = np.load(sentence_mv_path)

      sentence_dict = dict(zip(
        read_sentences(ordered_data_path), meaning_vectors))

      file = open(self._data_path(data_tag, 'txt'), 'r', encoding='utf-8')

      for index, line in enumerate(file):
        self.data_points[data_tag].append(self.DataPointClass(
          sentence_dict[line.strip('\n')], line, index, False))

      file.close()

    read('Source')
    read('Target')

    print("Finished reading " + self.tag + " data.")

  def _data_path(self, name, ext):
    return os.path.join(
        self.input_data_dir, self.tag + "{}.{}".format(name, ext))


def generate_encoder_states():

  # what hparams should we use
  if FLAGS["hparams"] == "":
    hparam_string = "general_" + FLAGS["model"] + "_hparams"
  else:
    hparam_string = FLAGS["hparams"]

  decode_mode_string = ""
  # determine the decode mode flag
  if FLAGS["decode_mode"] == "interactive":
    decode_mode_string = " --decode_interactive"
  elif FLAGS["decode_mode"] == "file":
    decode_mode_string = (" --decode_from_file="
                          + FLAGS["decode_dir"] + "/"
                          + FLAGS["input_file_name"])

  script_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'scripts', 'state_extraction.py')

  decode_file_path = FLAGS["decode_dir"] + "/" + FLAGS["output_file_name"]

  os.system("python3 {} \
                --generate_data=False \
                --t2t_usr_dir=".format(script_path) + FLAGS["t2t_usr_dir"]
            + " --data_dir=" + FLAGS["data_dir"]
            + " --problem=" + FLAGS["problem"]
            + " --output_dir=" + FLAGS["train_dir"]
            + " --model=" + FLAGS["model"]
            + " --worker_gpu_memory_fraction=" + str(FLAGS["memory_fraction"])
            + " --hparams_set=" + hparam_string
            + " --decode_to_file=" + decode_file_path
            + ' --decode_hparams="beam_size=' + str(FLAGS["beam_size"])
            + ",return_beams=" + FLAGS["return_beams"] + '"'
            + decode_mode_string)


def calculate_centroids(data_set, k, niter):
  if _use_faiss:
    verbose = True
    d = data_set.shape[1]
    kmeans = faiss.Kmeans(d, k, niter, verbose)
    kmeans.train(data_set)
    centroids = kmeans.centroids

  else:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_set)
    centroids = kmeans.cluster_centers_

  return centroids, kmeans


def calculate_nearest_index(data_point, kmeans):
  if _use_faiss:
    _, index = kmeans.index.search(data_point, 1)
  else:
    index = kmeans.predict(data_point)

  return index
