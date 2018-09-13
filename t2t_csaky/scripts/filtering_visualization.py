import os
import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import math
from collections import Counter
"""
This file contains functions for the same name jupyter notebook.
"""


# visualization function for the clustering data
def _visualize(file, tag, fig_list):
  """
  Params:
    :file: clustering file, from which to read data
    :tag: can be "Source" or "Target"
    :fig_list: a list containing the plots, which we will draw
  """
  sentence_entropy=[]
  entropies_all=[]
  entropies=[]
  sentence_cl_size=[]
  cl_sizes_all=[]
  cl_sizes=[]
  lengths=[]
  
  for line in file:
    [sentence, entropy, cl_size]=line.split(";")
    entropy=float(entropy)
    cl_size=int(cl_size)
    # use relative entropy
    #if entropy>0:
    #  entropy=entropy/math.log(cl_size, 2)

    # populate the lists
    sentence_entropy.append([sentence, entropy])
    entropies_all.extend([entropy]*cl_size)
    entropies.append(entropy)
    sentence_cl_size.append([sentence, cl_size])
    cl_sizes_all.extend([cl_size]*cl_size)
    cl_sizes.append(cl_size)
    lengths.append(len(sentence.split()))
  
  # draw the plots, and set properties
  fig_list[0].plot(sorted(entropies_all))
  fig_list[0].set_xlabel("Sentence no.")
  fig_list[0].set_ylabel("Entropy")
  fig_list[0].axis([0, 90000, -0.2, 10])

  fig_list[1].plot(sorted(cl_sizes_all))
  fig_list[1].set_xlabel("Sentence no.")
  fig_list[1].set_ylabel("Cluster size")
  # fig_list[1].axis([0, 90000, -0.2, 500])

  fig_list[2].scatter(np.array(cl_sizes), np.array(entropies))
  #fig_list[2].axis([0, 250, 7, 8])
  fig_list[2].set_xlabel("Cluster size")
  fig_list[2].set_ylabel("Entropy")

  fig_list[3].scatter(np.array(lengths), np.array(entropies))
  fig_list[3].set_xlabel("No. of words in utterance")
  fig_list[3].set_ylabel("Entropy")
  fig_list[3].axis([0, 50, 1.1, 8])
  
  # sort the sentence lists
  sent_ent=sorted(sentence_entropy, key=operator.itemgetter(1), reverse=True)
  sent_cl=sorted(sentence_cl_size, key=operator.itemgetter(1), reverse=True)
  return sent_ent, sent_cl

# main function to visualize clustering/filtering results
def data_visualization(source_cl,
                       target_cl,
                       dataset="DailyDialog",
                       cl_type="identity_clustering"):
  """
  Params:
    :source_cl: number of source clusters
    :target_cl: number of target clusters
    :dataset: name of the dataset
    :cl_type: type of the clustering method
  """
  # open the clustering files
  folder_name=cl_type+"/"+str(source_cl)+"-"+str(target_cl)+"_filtering/"
  source_cl_entropies=open(
    os.path.join("../../data_dir/"+dataset+"/base_with_numbers/filtered_data/"
                  +folder_name+"fullSource_cluster_entropies.txt"))
  target_cl_entropies=open(
    os.path.join("../../data_dir/"+dataset+"/base_with_numbers/filtered_data/"
                  +folder_name+"fullTarget_cluster_entropies.txt"))

  # set up matplotlib
  plt.close('all')
  fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6), (ax7,ax8))=plt.subplots(nrows=4,
                                                                 ncols=2)
  fig.set_size_inches(13, 20)

  # call the actual visualization function for source and target data
  source_entropies, source_cl_sizes=_visualize(source_cl_entropies,
                                               "Source",
                                               [ax1, ax3, ax5, ax7])
  target_entropies, target_cl_sizes=_visualize(target_cl_entropies,
                                               "Target",
                                               [ax2, ax4, ax6, ax8])
  plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
  
  source_cl_entropies.close()
  target_cl_entropies.close()
  return source_entropies, target_entropies, source_cl_sizes, target_cl_sizes

# reads the medoid and the corresponding sentences in the cluster
# TODO: refactor
def read_clusters():
  medoid_dict={}
  cluster_elements=open(
    "../../data_dir/"+dataset+"/base_with_numbers/filtered_data/hash_jaccard/"
    +str(source_clusters)+"_clusters/fullSource_cluster_elements.txt")
  for line in cluster_elements:
    line=line.split(";")[1]
    [medoid, pair, _]=line.split("<=====>")
    [source, target]=pair.split("=")
    if medoid in medoid_dict:
      medoid_dict[medoid].append(source)
    else:
      medoid_dict[medoid]=[source]


def print_clusters(source_cl, target_cl,
                   cl_type,
                   dataset='DailyDialog', tag='Source', top_k=12800):

  folder_name = cl_type + "/" + str(source_cl) + "-" + str(
    target_cl) + "_filtering/"

  clusters = {}
  cluster_element_lengths = {}

  with open(
    os.path.join(
      "../../data_dir/" + dataset + "/base_with_numbers/filtered_data/"
      + folder_name + "full{}_cluster_elements.txt".format(tag)), 'r') as file:

    for line in file:
      [source, source_cl_target, target_cl] = line.split('<=====>')

      if tag == 'Source':
        source_cl = source.split(';')[1]
        source = source_cl_target.split('=')[0]
        cluster_element_lengths[source_cl] = \
          cluster_element_lengths.get(source_cl, 0) + len(source.split())
        clusters[source_cl] = [*clusters.get(source_cl, []), source]

      else:
        target_cl = source.split(';')[1]
        target = source_cl_target.split('=')[0]
        cluster_element_lengths[target_cl] = \
          cluster_element_lengths.get(target_cl, 0) + len(target.split())
        clusters[target_cl] = [*clusters.get(target_cl, []), target]

  with open(
    os.path.join(
      "../../data_dir/" + dataset + "/base_with_numbers/filtered_data/"
      + folder_name +
              "full{}_cluster_entropies.txt".format(tag)), 'r') as file:

    entropies = {}
    for line in file:
      [medoid, entropy, size] = line.split(';')
      entropies[medoid] = float(entropy)

  for medoid in cluster_element_lengths:
    if ((cluster_element_lengths[medoid] /
      len(clusters[medoid]) if len(clusters[medoid]) > 0 else 1) > 20 or
            len(medoid.split()) > 20 or entropies[medoid] < 3.5):
      del clusters[medoid]

  for _, medoid in zip(range(top_k),
                       sorted(list(clusters), key=lambda x: entropies[x],
                              reverse=True)):
    print('=====================================================')
    print('Medoid: {} Entropy: {}'.format(medoid, entropies[medoid]))
    print('Size: {}'.format(len(clusters[medoid])))
    print('Elements: \n{}\n\n'.format('\n'.join(clusters[medoid])))

