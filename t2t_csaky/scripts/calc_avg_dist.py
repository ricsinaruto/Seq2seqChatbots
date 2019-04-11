import numpy as np
from random import sample
from scipy.spatial import distance

long_cluster = open("cluster_long.txt")
norm_cluster = open("cluster_normal.txt")
long_utts = [' '.join(line.split()) for line in long_cluster]
norm_utts = [' '.join(line.split()) for line in norm_cluster]
long_cluster.close()
norm_cluster.close()

full_source = dict([(' '.join(line.split()), i) for i, line in enumerate(open("fullSource.txt"))])
vectors = np.load('Source.npy')

norm_vectors = [vectors[full_source[utt]] for utt in norm_utts]
long_vectors = [vectors[full_source[utt]] for utt in long_utts]

distances = []
for i in range(100):
  vectors = sample(norm_vectors, 2)
  distances.append(distance.cosine(vectors[0], vectors[1]))

print(sum(distances) / len(distances))
