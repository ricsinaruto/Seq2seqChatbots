
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
  def __init__(self, string, index, only_string=True, meaning_vector=None):
    super().__init__(string, index, only_string)
    self.meaning_vector = meaning_vector


class RNNState(filter_problem.FilterProblem):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.decode_dir = FLAGS["decode_dir"]
    self.paths = {}

  @property
  def DataPointClass(self):
    return DataPoint

  def clustering(self, data_tag):
    meaning_vectors = np.load(self.paths[data_tag]['npy'])
    k = 3
    niter = 10

    centroids, kmeans = calculate_centroids(meaning_vectors, k, niter)
    clusters = [filter_problem.Cluster(centroid) for centroid in centroids]

    for data_point in self.data_points[data_tag]:
      cluster_index = calculate_nearest_index(
        data_point.meaning_vector, kmeans)

      clusters[cluster_index].add_element(data_point)
      data_point.cluster_index = cluster_index

    self.clusters[data_tag] = clusters

  # this function will read the data and make it ready for clustering
  def read_inputs(self):
    def read_sentences(file):
      sentences = []
      with open(file, 'r', encoding='utf-8') as f:
        for line in f:
          sentences.append(line.strip('\n'))
      return sentences

    def data_path(name, ext=''):
      return os.path.join(
        self.input_data_dir, self.tag + "{}.{}".format(name, ext))

    def read(data_tag):
      # if the encodings exists they will not be generated again
      # TODO meaning vector and decode output path fix
      # sentence meaning vector
      data_filter_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..')
      self.paths[data_tag] = {
        'txt': os.path.join(
                data_filter_path, self._decode_data_path(data_tag, 'txt')),
        'npy': os.path.join(
              data_filter_path, self._decode_data_path(data_tag, 'npy'))
      }

      if (not os.path.exists(self.paths[data_tag]['txt']) or
          not os.path.exists(self.paths[data_tag]['npy'])):

        generate_encoder_states(
          data_path(data_tag, 'txt'), '{}.txt'.format(data_tag))

      meaning_vectors = np.load(self.paths[data_tag]['npy'])

      sentence_dict = dict(zip(
        read_sentences(self.paths[data_tag]['txt']), meaning_vectors))

      file = open(data_path(data_tag, 'txt'), 'r',
                  encoding='utf-8')

      for index, line in enumerate(file):
        self.data_points[data_tag].append(self.DataPointClass(
          line, index, False, sentence_dict[line.strip('\n')]))

      file.close()

    read('Source')
    read('Target')

    print("Finished reading " + self.tag + " data.")

  def _decode_data_path(self, tag, ext=''):
    return os.path.join(
      self.decode_dir, '{}.{}'.format(tag, ext)
    )


def generate_encoder_states(input_file_path, output_file_name):
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
                          + input_file_path)

  script_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'scripts', 'state_extraction.py')

  os.system("python3 {} \
                --generate_data=False \
                --t2t_usr_dir=".format(script_path) + FLAGS["t2t_usr_dir"]
            + " --data_dir=" + FLAGS["data_dir"]
            + " --problem=" + FLAGS["problem"]
            + " --output_dir=" + FLAGS["train_dir"]
            + " --model=" + FLAGS["model"]
            + " --worker_gpu_memory_fraction=" + str(FLAGS["memory_fraction"])
            + " --hparams_set=" + hparam_string
            + " --decode_to_file=" + FLAGS["decode_dir"] + "/" +
            output_file_name
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
    index = kmeans.predict(data_point.reshape(1, -1))[0]

  return index
