import numpy as np

from bert_serving.client import BertClient
bc = BertClient()

def process_file(file, outfile):
  line_list = [line.strip('\n') for line in open(file, errors='ignore')]

  ndarray = bc.encode(line_list)
  np.save(outfile, ndarray)
  

process_file("fullSource.txt", "Source.npy")
process_file("fullSource.txt", "Target.npy")
