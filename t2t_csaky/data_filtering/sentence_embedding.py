"""

import numpy as np

source=open("devSource.txt")
#target=open("devTarget.txt")
vocab_file=open("vocab_dict.txt")

vocab_dict={}
# build vocab dict
for line in vocab_file:
  [word, embedding]=line.split(":")
  embeddings=embedding.strip("[").strip("]").split(", ").strip("'")
  array=np.array(range(len(embeddings)))
  for i in range(len(embeddings)):
    array[i]=float(embeddings[i])

  vocab_dict[word]=array

# a class to handle each data point
class Line:
  def __init__(self, string):
    self.string=string.strip("\n")
    self.ids=self.convert_to_ids(self.string)

  # convert string to vocab ids
  def convert_to_ids(self, string):
    words=string.split()
    ids=[]

    for word in words:
      if word in vocab_dict:
        ids.append(vocab_dict[word])
      else:
        ids.append(vocab_dict["<unk>"])

    return ids


# compute the distance of two embeddings
def emb_dist(word1, word2):
  return np.linalg.norm(word1-word2)

def process_weights():
  import numpy as np

  vocab=open("vocab.chatbot.16384")

  vocab_dict={}

  vocab_list=[]
  # read the vocab
  for word in vocab:
    vocab_list.append(word.strip("\n"))


  word_idx=0
  # read through the weight files
  for i in range(16):
    weight_file=open("weights/weights_"+str(i)+".txt")

    for embedding in weight_file:
      params=embedding.split(";")[:-1]

      # save to the vocab dict
      vocab_dict[vocab_list[word_idx]]=params
      word_idx+=1

    weight_file.close()

  # save the vocab dict to a file
  vocab_dict_file=open("vocab_dict.txt","w")

  for key in vocab_dict:
    vocab_dict_file.write(key+":")
    vocab_dict_file.write(str(vocab_dict[key]))
    vocab_dict_file.write("\n")

  vocab_dict_file.close()


  # do some processing


# distance metric between two sentences
def distance(first, second):
  # first delete the ones that are the same
  for word in first:
    if word in second:
      first.remove(word)
      second.remove(word)

  distances=[[] for x in range(len(word1))]
  # compute distance in one way
  i=0
  summ=0
  for word1 in first:
    minimum=1000000
    for word2 in second:
      dist=emb_dist(word1, word2)
      distances[i].append(dist)

      # compare
      if dist<minimum:
        minimum=dist
    i+=1
    summ+=minimum

  first_sum=summ/len(first)

  summ=0
  # compute distance in the other way
  for j in range(len(second)):
    minimum=1000000
    for i in range(len(first)):
      if distances[i][j]<minimum:
        minimum=distance[i][j]
    summ+=minimum

  second_sum=summ/len(second)

  return (first_sum+second_sum)/2


lines=[]
# parse the input file
for line in source:
  lines.append(Line(line))

"""
class SentenceEmbedding:
  def __init__():
    pass
