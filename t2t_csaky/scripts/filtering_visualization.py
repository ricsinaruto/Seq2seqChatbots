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
  fig_list[0].axis([78000, 90000, -0.2, 7.5])

  fig_list[1].plot(sorted(cl_sizes_all))
  fig_list[1].set_xlabel("Sentence no.")
  fig_list[1].set_ylabel("Cluster size")
  fig_list[1].axis([78000, 90000, -0.2, 10])

  fig_list[2].scatter(np.array(cl_sizes), np.array(entropies))
  fig_list[2].set_xlabel("Cluster size")
  fig_list[2].set_ylabel("Entropy")

  fig_list[3].scatter(np.array(lengths), np.array(entropies))
  fig_list[3].set_xlabel("No. of words in utterance")
  fig_list[3].set_ylabel("Entropy")
  fig_list[3].axis([0, 20, 1.1, 7])
  
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