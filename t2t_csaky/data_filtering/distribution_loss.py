import os
from collections import Counter

# My imports.
from config import DATA_FILTERING
from data_filtering.filter_problem import FilterProblem


# Calculate the target for each sentence in dataset.
class DistributionLoss(FilterProblem):

  # Recursive function to create probability matrix.
  def _create_tree(self, tr_word_matrix_tuple, tr_word_matrix, prob_matrix):
    # Copy the lists because of recursion.
    first_column = Counter(tr_word_matrix[0])

    # Calculate probabilities for the current column.
    for (row, index) in tr_word_matrix_tuple[0]:
      prob_matrix[index].append(first_column[row] / len(tr_word_matrix[0]))

    # Check stopping criterion.
    if len(tr_word_matrix) == 1:
      return [[1]]
    else:
      for distinct_word in first_column:
        indices = []
        # Get the rows which we want to continue the tree.
        for (word, index) in tr_word_matrix_tuple[0]:
          if word == distinct_word:
            indices.append(index)

        # Transponate to be able to delete rows.
        word_matrix = list(map(list, zip(*tr_word_matrix)))
        word_matrix_tuple = list(map(list, zip(*tr_word_matrix_tuple)))
        temp_mat = []
        temp_mat_tuple = []

        for row1, row2 in zip(word_matrix, word_matrix_tuple):
          if row2[0][1] in indices:
            temp_mat.append(row1)
            temp_mat_tuple.append(row2)

        # transponate back
        next_tr_matrix = list(map(list, zip(*temp_mat)))
        next_tr_matrix_tuple = list(map(list, zip(*temp_mat_tuple)))

        _ = self._create_tree(next_tr_matrix_tuple[1:],
                              next_tr_matrix[1:],
                              prob_matrix)
      return prob_matrix

  # Create input - target matrix pairs for distribution loss.
  def run(self):
    self.read_inputs()
    self.clustering("Source")
    self.clustering("Target")

    # Open data files.
    fSource = open(os.path.join(self.output_data_dir, "DLOSS_source.txt"), "w")
    fTarget = open(os.path.join(self.output_data_dir, "DLOSS_target.txt"), "w")

    # Loop through distinct inputs.
    for cl in self.clusters["Source"]:
      max_len = 0
      # Loop through the targets to get the longest.
      for target in cl.targets:
        sen_len = len(target.string.split())
        if DATA_FILTERING["max_length"] < sen_len:
          target.string = ""
        elif max_len < sen_len:
          max_len = sen_len

      word_matrix = []
      # Loop through targets to pad them and create a word matrix.
      for target in cl.targets:
        if target.string != "":
          words = target.string.split()
          word_matrix.append(words + ["<pad>"] * (max_len - len(words)))

      tr_word_matrix = list(map(list, zip(*word_matrix)))
      prob_matrix = [[] for row in word_matrix]

      # Add row indices to word matrix.
      for i, target in enumerate(word_matrix):
        for j, word in enumerate(target):
          word_matrix[i][j] = (word, i)
      tr_word_matrix_tuple = list(map(list, zip(*word_matrix)))

      # Recurse to create tree.
      if word_matrix != []:
        fSource.write(cl.medoid.string + "\n")
        prob_matrix = self._create_tree(tr_word_matrix_tuple,
                                        tr_word_matrix,
                                        prob_matrix)

      # Save target matrix to file.
      for target_words, target_probs in zip(word_matrix, prob_matrix):
        for (word, index), prob in zip(target_words, target_probs):
          fTarget.write(word + ":" + str(prob) + " ")
        fTarget.write("\n")
      fTarget.write("\n")

    fSource.close()
    fTarget.close()
