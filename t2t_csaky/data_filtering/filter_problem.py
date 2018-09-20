import os
import math
from collections import Counter

# My imports.
from config import FLAGS, DATA_FILTERING, PROBLEM_HPARAMS


class DataPoint:
  """
  A simple class that handles a string example.
  """
  def __init__(self, string, index, only_string=True):
    """
    Params:
      :string: String to be stored.
      :index: Number of the line in the file from which this sentence was read.
      :only_string: Whether to only store string.
    """
    self.index = index
    self.string = string.strip("\n")
    self.cluster_index = 0


class Cluster:
  """
  A class to handle one cluster in the clustering problem.
  """
  def __init__(self, medoid):
    """
    Params:
      :medoid: Center of the cluster: a data point object.
    """
    self.medoid = medoid
    self.elements = []
    self.targets = []
    self.entropy = 0
    self.index = 0

  # Append an element to the list of elements in the cluster.
  def add_element(self, element):
    self.elements.append(element)

  # append an element to the list of targets in the cluster.
  def add_target(self, target):
    self.targets.append(target)


class FilterProblem:
  """
  An abstract class to handle different types of filtering.
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
      :tag: Can be either full, train, dev, or test.
    """
    self.tag = tag
    self.treshold = DATA_FILTERING["treshold"]
    self.max_avg_length = DATA_FILTERING["max_avg_length"]
    self.max_medoid_length = DATA_FILTERING["max_medoid_length"]
    self.min_cluster_size = DATA_FILTERING["min_cluster_size"]

    self.project_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..')
    self.output_data_dir = DATA_FILTERING["data_dir"]
    self.input_data_dir = FLAGS["data_dir"]
    self.type = DATA_FILTERING["filter_type"]

    self.clusters = {"Source": [], "Target": []}
    self.data_points = {"Source": [], "Target": []}
    self.num_clusters = {"Source": 0, "Target": 0}

    # Extra step to figure out in which split to put the results.
    train_lines = self.count_lines("train")
    dev_lines = self.count_lines("dev")
    test_lines = self.count_lines("test")

    self.split_line_counts = {
        "train": train_lines,
        "dev": dev_lines,
        "test": test_lines
    }

    # Calculate number of clusters.
    if self.tag == "train" or self.tag == "full":
      self.num_clusters["Source"] = DATA_FILTERING["source_clusters"]
      self.num_clusters["Target"] = DATA_FILTERING["target_clusters"]
    else:
      self.num_clusters["Source"] = \
          int(DATA_FILTERING["source_clusters"] *
              PROBLEM_HPARAMS["dataset_split"][self.tag] /
              PROBLEM_HPARAMS["dataset_split"]["train"])
      self.num_clusters["Target"] = \
          int(DATA_FILTERING["target_clusters"] *
              PROBLEM_HPARAMS["dataset_split"][self.tag] /
              PROBLEM_HPARAMS["dataset_split"]["train"])

  # Count the number of lines in the given file.
  def count_lines(self, file_tag="train"):
    file = open(os.path.join(self.project_path,
                             self.input_data_dir,
                             file_tag + "Source.txt"))
    num_lines = sum(1 for line in file)
    file.close()
    return num_lines

  # Main method that will run all the functions to do the filtering.
  def run(self):
    # If we have already done the clustering, don't redo it.
    source_data = os.path.join(self.project_path, self.output_data_dir,
                               str(self.num_clusters["Source"]) + '-' +
                               str(self.num_clusters["Target"]) + '_filtering',
                               self.tag + "Source_cluster_elements.txt")

    target_data = os.path.join(self.project_path, self.output_data_dir,
                               str(self.num_clusters["Source"]) + '-' +
                               str(self.num_clusters["Target"]) + '_filtering',
                               self.tag + "Target_cluster_elements.txt")
    if os.path.isfile(source_data) and os.path.isfile(target_data):
      print("Cluster files are in " + self.output_data_dir +
            ", filtering now.")
      self.load_clusters_merged()
      self.filtering()

    else:
      print("No cluster files in " + self.output_data_dir +
            ", clustering now.")
      self.read_inputs()
      self.clustering("Source")
      self.clustering("Target")
      self.save_clusters("Source")
      self.save_clusters("Target")
      self.filtering()

  # This function will read the data and make it ready for clustering.
  def read_inputs(self):
    # Read either a Source or target file.
    def read_file(tag):
      file = open(os.path.join(self.input_data_dir, self.tag + tag))
      for i, line in enumerate(file):
        self.data_points[tag].append(self.DataPointClass(line, i, False))
      file.close()

    read_file("Source")
    read_file("Target")
    print("Finished reading " + self.tag + " data.")

  # Load clusters from files.
  def load_clusters(self):
    # Open the data files.
    source_clusters = os.path.join(
        self.project_path, self.output_data_dir,
        str(self.num_clusters["Source"]) + '-' +
        str(self.num_clusters["Target"]) + '_filtering',
        self.tag + "Source_cluster_elements.txt")

    target_clusters = os.path.join(
        self.project_path, self.output_data_dir,
        str(self.num_clusters["Source"]) + '-' +
        str(self.num_clusters["Target"]) + '_filtering',
        self.tag + "Target_cluster_elements.txt")

    # Make a preloaded target cluster list.
    self.clusters["Target"] = ["" for i in range(self.num_clusters["Target"])]
    target_cluster_list = ["" for i in range(self.split_line_counts["train"] +
                                             self.split_line_counts["dev"] +
                                             self.split_line_counts["test"])]
    # Read the target clusters first.
    for line in target_clusters:
      [index, line] = line.split(";")
      [source_medoid, pair, target_cluster] = line.strip("\n").split("<=====>")
      # List containing target medoid and target cluster index.
      [target_medoid, target_cl_index] = target_cluster.split(":")
      target_cluster_list[int(index)] = [target_medoid, int(target_cl_index)]

    # Load the source clusters.
    last_medoid = "<-->"
    for line in source_clusters:
      [index, line] = line.split(";")
      [source_medoid, pair, target_cluster] = line.strip("\n").split("<=====>")
      [source, target] = pair.split("=")
      [target_medoid, target_cl_index] = target_cluster_list[int(index)]
      index = int(index)

      source_data_point = self.DataPointClass(source, index, True)
      target_data_point = self.DataPointClass(target, index, True)
      source_data_point.cluster_index = len(self.clusters["Source"])
      target_data_point.cluster_index = target_cl_index

      # Check if this is a new medoid (source side).
      if last_medoid != source_medoid:
        # Add medoid to cluster.
        dp = self.DataPointClass(source_medoid, index=0, only_string=True)
        self.clusters["Source"].append(self.ClusterClass(dp))
      self.clusters["Source"][-1].add_element(source_data_point)
      self.clusters["Source"][-1].add_target(target_data_point)

      # Target side.
      if self.clusters["Target"][target_cl_index] == "":
        dp = self.DataPointClass(target_medoid, index=0, only_string=True)
        self.clusters["Target"][target_cl_index] = self.ClusterClass(dp)
      self.clusters["Target"][target_cl_index].add_element(target_data_point)
      self.clusters["Target"][target_cl_index].add_target(source_data_point)

      last_medoid = source_medoid

    source_clusters.close()
    target_clusters.close()

  # Load clusters that were saved in a different way (semantic clustering).
  def load_clusters_merged(self):
    source_path = os.path.join(
        self.project_path, self.output_data_dir,
        str(self.num_clusters["Source"]) + '-' +
        str(self.num_clusters["Target"]) + '_filtering',
        self.tag + "Source_cluster_elements.txt")

    target_path = os.path.join(
        self.project_path, self.output_data_dir,
        str(self.num_clusters["Source"]) + '-' +
        str(self.num_clusters["Target"]) + '_filtering',
        self.tag + "Target_cluster_elements.txt")

    source_clusters = {}
    target_clusters = {}

    source_data_points = {}
    target_data_points = {}

    with open(source_path, 'r') as source_file:
      for line in source_file:
        [source_index_center, source_target, _] = line.split('<=====>')
        [source_index, source_center] = source_index_center.split(';')
        [source, target] = source_target.split('=')

        source_data_points[int(source_index)] = self.DataPointClass(
            source, int(source_index), False)
        target_data_points[int(source_index)] = self.DataPointClass(
            target, int(source_index), False)

        if source_clusters.get(source_center) is None:
          center = self.DataPointClass(source_center, 0, False)
          source_clusters[source_center] = self.ClusterClass(center)
          source_clusters[source_center].index = len(source_clusters) - 1

        source_data_points[int(source_index)].cluster_index = \
            source_clusters[source_center].index
        source_clusters[source_center].add_element(
            source_data_points[int(source_index)])
        source_clusters[source_center].add_target(
            target_data_points[int(source_index)])

    with open(target_path, 'r') as target_file:
      for line in target_file:
        [target_index_center, target_source, _] = line.split('<=====>')
        [target_index, target_center] = target_index_center.split(';')
        [target, source] = target_source.split('=')

        target_data_point = target_data_points[int(target_index)]
        source_data_point = source_data_points[int(target_index)]

        if target_clusters.get(target_center) is None:
          center = self.DataPointClass(target_center, 0, False)
          target_clusters[target_center] = self.ClusterClass(center)
          target_clusters[target_center].index = len(target_clusters) - 1

        target_data_point.cluster_index = target_clusters[target_center].index
        target_clusters[target_center].add_element(target_data_point)
        target_clusters[target_center].add_target(source_data_point)

    self.clusters['Source'] = list(source_clusters.values())
    self.clusters['Target'] = list(target_clusters.values())

  # Find the point that minimizes mean distance within a cluster.
  def find_medoid(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data.
    """
    for cluster in self.clusters[data_tag]:
      print("Finding medoids.")
      big_sum = 0
      for element1 in cluster.elements:
        small_sum = 0
        for element2 in cluster.elements:
          small_sum += element1.similarity(element2, self.dist_matrix)

        if small_sum > big_sum:
          big_sum = small_sum
          cluster.medoid = element1

      # Clear elements after we finished with one cluster.
      cluster.elements.clear()
      cluster.targets.clear()

  # Find nearest medoid for a data point.
  def find_nearest_medoid(self, data_point):
    pass

  # For each data_point find a cluster.
  def cluster_points(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data.
    """
    # Reverse data tag.
    rev_tag = "Target" if data_tag == "Source" else "Source"

    for i, data_point in enumerate(self.data_points[data_tag]):
      nearest_medoid = self.find_nearest_medoid(data_point, data_tag)
      self.clusters[data_tag][nearest_medoid].add_element(data_point)
      self.clusters[data_tag][nearest_medoid].add_target(
          self.data_points[rev_tag][i])

      data_point.cluster_index = nearest_medoid

  # A function that checks if clustering needs to stop.
  def stop_clustering(self, data_tag, clusters, clusters_old, count, counts):
    """
    Params:
      :data_tag: Whether it's source or target data.
      :clusters: String of medoids from previous iteration.
      :clusters_old: String of medoids from 2 iterations ago.
      :count: Number of clustering loops so far.
    """
    count_difference = 0
    count_difference_old = 0
    for i, cluster in enumerate(self.clusters[data_tag]):
      # Check strings from previous iteration to see if they are the same.
      if cluster.medoid.string != clusters[i]:
        count_difference += 1
        print(clusters[i] + "--->" + cluster.medoid.string)
        clusters[i] = cluster.medoid.string

      # Check strings from two loops ago, to see if they are the same.
      if cluster.medoid.string != clusters_old[i]:
        count_difference_old += 1
        if count % 2 == 0:
          clusters_old[i] = cluster.medoid.string
    print("==================================================")
    print("==================================================")

    # Check if no. of medoids changed is the same for the last 6 iterations.
    same_counts = True
    counts.append(count_difference)
    if len(counts) > 6:
      counts = list(counts[1:])
    for i, c in enumerate(counts[:-1]):
      if c != counts[i + 1]:
        same_counts = False

    # Exit if there is no change or we are stuck in a loop.
    exit = False
    if count_difference == 0 or count_difference_old == 0 or same_counts:
      exit = True
    return exit, clusters, clusters_old, counts

  # Do the clustering of sources and targets.
  def clustering(self, data_tag):
    pass

  # Return a list of indices, showing which clusters should be filtered out.
  def get_filtered_indices(self, source):
    """
    Params:
      :source: The cluster that we want to filter (either Source or Target).
    """
    indices = []
    for num_cl, cluster in enumerate(self.clusters[source]):
      # Error guarding for the case when loading clusters.
      if cluster != "":
        # Build a distribution for the current cluster, based on the targets.
        distribution = Counter()
        for target in cluster.targets:
          if target.cluster_index in distribution:
            distribution[target.cluster_index] += 1
          else:
            distribution[target.cluster_index] = 1

        num_elements = len(cluster.elements)
        # Calculate entropy.
        entropy = 0
        for cl_index in distribution:
          if num_elements > 1:
            probability = distribution[cl_index] / num_elements
            entropy += probability * math.log(probability, 2)
        cluster.entropy = -entropy

        avg_length = (
            sum(len(sent.string.split()) for sent in cluster.elements) /
            (num_elements if num_elements > 0 else 1))

        # Filter.
        if (cluster.entropy > self.treshold and
            avg_length < self.max_avg_length and
                len(cluster.medoid.string.split()) < self.max_medoid_length):
          indices.append(num_cl)
          print('Medoid: "' + cluster.medoid.string + '" got filtered.')

    print("Finished filtering " + source + " data.")
    return indices

  # Do the filtering of the dataset.
  def filtering(self):
    # These are not needed anymore.
    self.data_points["Source"].clear()
    self.data_points["Target"].clear()

    # Get the filtered indices for both sides.
    source_indices = self.get_filtered_indices("Source")
    target_indices = self.get_filtered_indices("Target")

    # Open files, where data will be written.
    file_dict = {}
    # We have to open 6 files in this case.
    if self.tag == "full":
      name_list = ["trainS", "trainT", "devS", "devT", "testS", "testT"]
      file_list = list(self.open_6_files())
      file_dict = dict(zip(name_list, file_list))

    # Handle all cases and open files.
    if self.type == "target_based" or self.type == "both":
      file_dict["source_entropy"] = open(
          os.path.join(self.output_data_dir,
                       self.tag + "Source_cluster_entropies.txt"), "w")
    if self.type == "source_based" or self.type == "both":
      file_dict["target_entropy"] = open(
          os.path.join(self.output_data_dir,
                       self.tag + "Target_cluster_entropies.txt"), "w")
    file_dict[self.tag + "source_file"] = open(
        os.path.join(self.output_data_dir, self.tag + "Source.txt"), "w")
    file_dict[self.tag + "target_file"] = open(
        os.path.join(self.output_data_dir, self.tag + "Target.txt"), "w")

    # Save data and close files.
    self.save_filtered_data(source_indices, target_indices, file_dict)
    self.close_n_files(file_dict)

  # Save the new filtered datasets.
  def save_filtered_data(self, source_indices, target_indices, file_dict):
    """
    Params:
      :source_indices: Indices of source clusters that will be filtered.
      :target_indices: Indices of target clusters that will be filtered.
      :file_dict: Dictionary containing all the files that we want to write.
    """
    # Function for writing the dataset to file.
    def save_dataset(source):
      for num_cl, cluster in enumerate(self.clusters[source]):
        # Filter errors due to cluster loading.
        if cluster != "":
          # Write cluster entropies.
          file_dict[source.lower() + "_entropy"].write(
              cluster.medoid.string + ";" +
              str(cluster.entropy) + ";" +
              str(len(cluster.elements)) + "\n")

          indices = source_indices if source == "Source" else target_indices
          cluster_too_small = len(cluster.elements) < self.min_cluster_size
          # Make sure that in "both" case this is only run once.
          if ((source == "Source" or self.type != "both") and
                  (num_cl not in indices or cluster_too_small)):
            # Filter one side.
            for num_el, element in enumerate(cluster.elements):
              target_cl = cluster.targets[num_el].cluster_index
              if self.type == "both":
                cluster_too_small = (len(
                    self.clusters["Target"][target_cl].elements) <
                    self.min_cluster_size)
              # Check both sides in "both" case.
              if ((target_cl not in target_indices or cluster_too_small) or
                      self.type != "both"):
                source_string = element.string + "\n"
                target_string = cluster.targets[num_el].string + "\n"

                # Reverse if Target.
                if source == "Target":
                  tmp = source_string
                  source_string = target_string
                  target_string = tmp
                file_dict[self.tag + "source_file"].write(source_string)
                file_dict[self.tag + "target_file"].write(target_string)

                # Write to separate files if we do split after clustering.
                if self.tag == "full":
                  if element.index < self.split_line_counts["train"]:
                    file_dict["trainS"].write(source_string)
                    file_dict["trainT"].write(target_string)
                  elif element.index < (self.split_line_counts["train"] +
                                        self.split_line_counts["dev"]):
                    file_dict["devS"].write(source_string)
                    file_dict["devT"].write(target_string)
                  else:
                    file_dict["testS"].write(source_string)
                    file_dict["testT"].write(target_string)

    # Write source entropies and data to file.
    if self.type == "target_based" or self.type == "both":
      save_dataset("Source")
    # Write target entropies and data to file.
    if self.type == "source_based" or self.type == "both":
      save_dataset("Target")

  # Save clusters and their elements to files.
  def save_clusters(self, data_tag):
    """
    Params:
      :data_tag: Whether it's source or target data.
    """
    output = open(
        os.path.join(self.output_data_dir,
                     self.tag + data_tag + "_cluster_elements.txt"), "w")
    medoid_counts = []
    rev_tag = "Target" if data_tag == "Source" else "Source"

    for cluster in self.clusters[data_tag]:
      medoid_counts.append((cluster.medoid.string, len(cluster.elements)))

      # Save together the source and target medoids and elements.
      for source, target in zip(cluster.elements, cluster.targets):
        output.write(
            str(source.index) + ";" +
            cluster.medoid.string + "<=====>" +
            source.string + "=" +
            target.string + "<=====>" +
            self.clusters[rev_tag][target.cluster_index].medoid.string + ":" +
            str(target.cluster_index) + "\n")
    output.close()

    # Save the medoids and the count of their elements, in decreasing order.
    output = open(os.path.join(self.output_data_dir,
                               self.tag + data_tag + "_clusters.txt"), "w")
    medoids = sorted(medoid_counts, key=lambda count: count[1], reverse=True)

    for medoid in medoids:
      output.write(medoid[0] + ":" + str(medoid[1]) + "\n")
    output.close()

    print("Finished clustering, proceeding with filtering.")

  # Open the 6 files.
  def open_6_files(self):
    trainS = open(os.path.join(self.output_data_dir, 'trainSource.txt'), 'w')
    trainT = open(os.path.join(self.output_data_dir, 'trainTarget.txt'), 'w')
    devS = open(os.path.join(self.output_data_dir, 'devSource.txt'), 'w')
    devT = open(os.path.join(self.output_data_dir, 'devTarget.txt'), 'w')
    testS = open(os.path.join(self.output_data_dir, 'testSource.txt'), 'w')
    testT = open(os.path.join(self.output_data_dir, 'testTarget.txt'), 'w')

    return trainS, trainT, devS, devT, testS, testT

  # Close n files to write the processed data into.
  def close_n_files(self, files):
    for file_name in files:
      files[file_name].close()
