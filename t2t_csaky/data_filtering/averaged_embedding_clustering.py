import os
import numpy as np


import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

# my imports
from semantic_clustering import SemanticClustering


class AverageWordEmbedding(SemanticClustering):
  """
  Averaged word embeddings clustering method. The meaning vector of the
  sentence is created by the weighted average of the word vectors.
  """

  def _read(self, data_tag):
    # if the encodings exists they will not be generated again
    # sentence meaning vector
    project_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), '..', '..')

    self.paths[data_tag] = {'npy': os.path.join(
      project_path, self._data_path(data_tag, '.npy'))}

    if not os.path.exists(self.paths[data_tag]['npy']):
      self.generate_average_word_embeddings(
        os.path.join(project_path,
                     self._data_path('vocab')),
        self._data_path(self.tag + data_tag, '.txt'),
        self.paths[data_tag])

    meaning_vectors = np.load(self.paths[data_tag]['npy'])

    file = open(self._data_path(self.tag + data_tag, '.txt'), 'r',
                encoding='utf-8')

    for index, line in enumerate(file):
      self.data_points[data_tag].append(self.DataPointClass(
        line.strip(), index, False, meaning_vectors[index]))

    file.close()

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
        vocab[line_as_list[0]] = [0,
          np.array([float(num) for num in line_as_list[1:]])]

    embedding_dim = len(vocab[list(vocab)[0]][1])

    word_count = 0
    with open(input_file_path, 'r') as f:
      for line in f:
        line_as_list = line.strip().split()
        for word in line_as_list:

          vector = vocab.get(word)
          if vector is not None:
            vocab[word][0] += 1
            word_count += 1

    meaning_vectors = []
    with open(input_file_path, 'r') as f:
      for line in f:
        line_as_list = line.strip().split()
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
