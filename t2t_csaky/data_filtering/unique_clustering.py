import os
import numpy as np
from sklearn.neighbors import BallTree

# My imports.
from config import DATA_FILTERING
from utils.utils import calculate_centroids_mean_shift
from utils.utils import calculate_centroids_kmeans
from data_filtering.average_word_embedding import AverageWordEmbedding


class UniqueClustering(AverageWordEmbedding):
  """
  Averaged word embeddings clustering method. The meaning vector of the
  sentence is created by the weighted average of the word vectors.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # These will hold the actual unique sentences to be clustered.
    self.unique_data = {"Source": [], "Target": []}
    self.meaning_vectors = {"Source": None, "Target": None}

  def clustering(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data.
    """
    data_point_vectors = np.array(
        [data_point.meaning_vector for data_point in
         self.unique_data[data_tag]])
    data_point_vectors = data_point_vectors.reshape(
        -1, self.unique_data[data_tag][0].meaning_vector.shape[-1])

    if DATA_FILTERING["semantic_clustering_method"] == "mean_shift":
      centroids, method, = calculate_centroids_mean_shift(
          self.meaning_vectors[data_tag])

    else:
      n_clusters = DATA_FILTERING['{}_clusters'.format(data_tag.lower())]
      centroids, method, = calculate_centroids_kmeans(
        self.meaning_vectors[data_tag], niter=20, n_clusters=n_clusters)

    tree = BallTree(data_point_vectors)
    _, centroids = tree.query(centroids, k=1)
    tree = BallTree(data_point_vectors[np.array(centroids).reshape(-1)])
    _, labels = tree.query(data_point_vectors, k=1)

    labels = labels.reshape(-1)

    clusters = {index: self.ClusterClass(self.unique_data[data_tag][index]) for
                index in {labels[_index] for _index in range(len(labels))}}

    clusters = [(clusters[cluster_index], cluster_index) for cluster_index in
                sorted(list(clusters))]

    label_lookup = {c[1]: i for i, c in enumerate(clusters)}
    clusters = [c[0] for c in clusters]

    rev_tag = "Target" if data_tag == "Source" else "Source"

    # Store the cluster index for each unique sentence.
    cluster_ind_dict = {}
    for data_point, cluster_index in zip(self.unique_data[data_tag], labels):
      cluster_ind_dict[data_point.string] = label_lookup[cluster_index]

    # Get all sentences from the unique
    for i, data_point in enumerate(self.data_points[data_tag]):
      cl_index = cluster_ind_dict[data_point.string]
      data_point.cluster_index = cl_index

      # Put the data point in the respective cluster.
      clusters[cl_index].add_element(data_point)
      clusters[cl_index].add_target(self.data_points[rev_tag][i])

    self.clusters[data_tag] = clusters

  def _read(self, data_tag):
    # If the encodings exists they will not be generated again.
    project_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..')

    self.paths[data_tag] = {'npy': os.path.join(
        project_path, self._data_path(data_tag, '.npy'))}

    if not os.path.exists(self.paths[data_tag]['npy']):
      self.generate_average_word_embeddings(
          os.path.join(project_path,
                       self._data_path('vocab.chatbot.16384')),
          self._data_path(self.tag + data_tag, '.txt'),
          self.paths[data_tag]['npy'])

    self.meaning_vectors[data_tag] = np.load(self.paths[data_tag]['npy'])

    # Unique set of sentences.
    unique_sentences = set()
    with open(self._data_path(
            self.tag + data_tag, '.txt'), 'r', encoding='utf-8') as file:
      for index, line in enumerate(file):
        self.data_points[data_tag].append(self.DataPointClass(
            line.strip(), index, False))
        unique_sentences.add(line.strip())

    # Create the unique data points.
    for i, sent in enumerate(unique_sentences):
      self.unique_data[data_tag].append(self.DataPointClass(
          sent, i, False, self.meaning_vectors[data_tag][i]))

  def generate_average_word_embeddings(self,
                                       vocab_path,
                                       input_file_path,
                                       output_file_path):
    """
    Generates the word embeddings for a given file.

    Params:
      :vocab_path: A path of the vocabulary file, that should contain
                   word - vector pairs, where the word, and each
                   number is separated by a single space.
    """
    vocab = {}
    with open(vocab_path, 'r') as v:
      for line in v:
        line_as_list = line.strip().split()
        vocab[line_as_list[0]] = [
            0, np.array([float(num) for num in line_as_list[1:]])]

    embedding_dim = len(vocab[list(vocab)[0]][1])

    # Save unique sentences into a set.
    unique_sentences = set()
    word_count = 0
    with open(input_file_path, 'r') as f:
      for line in f:
        unique_sentences.add(line.strip())
        line_as_list = line.strip().split()
        for word in line_as_list:

          vector = vocab.get(word)
          if vector is not None:
            vocab[word][0] += 1
            word_count += 1

    meaning_vectors = []
    for sent in unique_sentences:
      line_as_list = sent.split()

      vectors = []
      for word in line_as_list:
        vector = vocab.get(word)
        if vector is not None:
          vectors.append(vector[1] * 0.001 /
                         (0.001 + vector[0] / word_count))

      if len(vectors) == 0:
        meaning_vectors.append(np.zeros(embedding_dim))
      else:
        meaning_vectors.append(np.sum(np.array(vectors), axis=0) /
                               len(vectors))

    np.save(output_file_path, np.array(meaning_vectors).
            reshape(-1, embedding_dim))
