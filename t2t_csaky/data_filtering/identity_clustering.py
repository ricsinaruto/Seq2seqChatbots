
# My imports.
from data_filtering.filter_problem import FilterProblem


class IdentityClustering(FilterProblem):
  """
  Only cluster sentences that are exactly the same.
  """

  # Do the clustering of sources and targets.
  def clustering(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data.
    """
    # Reverse data tag.
    rev_tag = "Target" if data_tag == "Source" else "Source"

    sentence_list = [" ".join(dp.string.split())
                     for dp in self.data_points[data_tag]]
    sentence_set = set(sentence_list)
    sentence_set = list(sentence_set)

    # Build a hash for efficient string searching.
    sentence_dict = {}
    for data_point in self.data_points[data_tag]:
      clean_sentence = " ".join(data_point.string.split())
      if clean_sentence in sentence_dict:
        sentence_dict[clean_sentence].append(data_point)
      else:
        sentence_dict[clean_sentence] = [data_point]

    print(data_tag + ": " + str(len(sentence_set)) + " clusters")

    # Loop through the clusters.
    for i, sentence in enumerate(sentence_set):
      cl = self.ClusterClass(self.DataPointClass(sentence, 10))
      self.clusters[data_tag].append(cl)

      # Loop through the dataset.
      for data_point in sentence_dict[sentence]:
        data_point.cluster_index = i
        cl.add_element(data_point)
        cl.targets.append(self.data_points[rev_tag][data_point.index])
