# https://github.com/julianser/hed-dlg-truncated
import math
import numpy as np
from scipy.spatial import distance


# Paths to the different data files.
train_source_path = "data_dir/DailyDialog/base_with_numbers/trainSource.txt"
test_responses_path = "decode_dir/DailyDialog/trf_20_dropout-base_both_avg_embedding/test_set_13k.txt"
gt_responses_path = "data_dir/DailyDialog/base_with_numbers/testTarget.txt"
test_source_path = "data_dir/DailyDialog/base_with_numbers/testSource.txt"
text_vocab_path = "data_dir/DailyDialog/base_with_numbers/vocab.chatbot.16384"
vector_vocab_path = "data_dir/DailyDialog/base_with_numbers/vocab.chatbot.16384_vector"

# Some globals.
output = open(test_responses_path.strip(".txt") + "_metrics.txt", "w")
vocab = {}


# Count words and load the vocab files.
def count_words():
  train_corpus = open(train_source_path)
  embeddings = open(vector_vocab_path)
  vocab_file = open(text_vocab_path)
  num_words = 0

  # Build the word vectors.
  for line in embeddings:
    line_as_list = line.strip().split()
    vocab[line_as_list[0]] = [
        0, np.array([float(num) for num in line_as_list[1:]])]

  emb_dim = len(vocab[list(vocab)[0]][1])

  # Extend the remaining vocab.
  for line in vocab_file:
    if line.strip() not in vocab.keys():
      vocab[line.strip()] = [0, np.zeros(emb_dim)]

  # Go through the train file and build word frequencies.
  for line in train_corpus:
    for word in line.strip("\n").split():

      num_words += 1
      if vocab.get(word) is not None:
        vocab[word][0] += 1
      else:
        vocab["<unk>"][0] += 1

  train_corpus.close()
  embeddings.close()
  vocab_file.close()
  return num_words, emb_dim


# Compute mean, std and confidence, and write the given metric to file.
def write_metric(metric, name):
  """
  Params:
    :metric: A list of numbers representing a metric for each response.
    :name: Name of the metric.
  """
  avg = sum(metric) / len(metric)
  std = 0
  if len(metric) > 1:
    std = np.std(metric)

  # 95% confidence interval (t=1.97)
  conf = 1.97 * std / math.sqrt(len(metric))

  # Write the metric to file.
  m = name + ": " + str(avg) + " " + str(std) + " " + str(conf)
  print(m)
  output.write(m + "\n")


# Calculate the embedding average metric.
def calculate_avg_embedding(gt_words, test_words, num_words, emb_dim):
  def sentence_vector(words):
    vectors = []
    for word in words:
      vector = vocab.get(word)
      if vector is not None:
        vectors.append(vector[1] * 0.001 /
                      (0.001 + vector[0] / num_words))

    if len(vectors):
      return np.sum(np.array(vectors), axis=0) / len(vectors)
    else:
      return np.zeros(emb_dim)

  # Compute cosine similarity.
  gt = sentence_vector(gt_words)
  test = sentence_vector(test_words)
  zeros = np.zeros(emb_dim)
  if np.all(gt == zeros) or np.all(test == zeros):
    return None
  else:
    return 1 - distance.cosine(gt, test)


# Calculate the embedding extrema metric.
def calculate_extrem_embedding(gt_words, test_words, num_words, emb_dim):
  def sentence_vector(words):
    vector = np.zeros(emb_dim)
    for word in words:
      vec = vocab.get(word)
      if vec is not None:
        for i in range(emb_dim):
          if abs(vec[1][i]) > abs(vector[i]):
            vector[i] = vec[1][i]
    return vector

  # Compute cosine similarity.
  gt = sentence_vector(gt_words)
  test = sentence_vector(test_words)
  zeros = np.zeros(emb_dim)
  if np.all(gt == zeros) or np.all(test == zeros):
    return None
  else:
    return 1 - distance.cosine(gt, test)


# Calculate the embedding greedy metric.
def calculate_greedy_embedding(gt_words, test_words, num_words, emb_dim):
  def score(one, two):
    y_vec = np.zeros((emb_dim, 1))
    x_count = 0
    y_count = 0
    cos_sim = 0
    for word in two:
      vec = vocab.get(word)
      if vec is not None:
        norm = np.linalg.norm(vec[1])
        vec = vec[1] / norm if norm else vec[1]
        y_vec = np.hstack((y_vec, (vec.reshape((emb_dim, 1)))))
        y_count += 1

    for word in one:
      vec = vocab.get(word)
      if vec is not None:
        norm = np.linalg.norm(vec[1])
        vec = vec[1] / norm if norm else vec[1]
        cos_sim += np.max(vec.reshape((1, emb_dim)).dot(y_vec))
        x_count += 1

    if x_count > 0 and y_count > 0:
      return cos_sim / x_count
    else:
      return 0

  return (score(gt_words, test_words) + score(test_words, gt_words)) / 2


# Compute all metrics based on test responses.
def metrics(num_words, emb_dim):
  """
  Params:
    :num_words: Number of words in the train file.
    :emb_dim: Dimensions size of word embeddings.
  """
  test_responses = open(test_responses_path)
  gt_responses = open(gt_responses_path)
  entropies = []
  response_len = []
  utt_entropy = []
  embedding = {"avg": [], "extrem": [], "greedy": []}
  word_set = set()
  bigrams = set()
  test_words = 0

  # Loop through the test and ground truth responses, and calculate metrics.
  for test_line, gt_line in zip(test_responses, gt_responses):
    gt_words = gt_line.strip("\n").split()
    words = test_line.strip("\n").split()
    word_count = len(words)
    response_len.append(word_count)
    test_words += word_count

    # Calculate embedding metrics.
    avg = calculate_avg_embedding(gt_words, words, num_words, emb_dim)
    extrem = calculate_extrem_embedding(gt_words, words, num_words, emb_dim)
    greedy = calculate_greedy_embedding(gt_words, words, num_words, emb_dim)
    if avg is not None:
      embedding["avg"].append(avg)
    if extrem is not None:
      embedding["extrem"].append(extrem)
    if greedy is not None:
      embedding["greedy"].append(greedy)

    # Calculate entropy metrics.
    entropy = 0
    for i, word in enumerate(words):
      word_set.add(word)
      if i < word_count - 1:
        bigrams.add(word + words[i + 1])

      # Calculate word entropy.
      if vocab.get(word) is not None:
        probability = vocab[word][0] / num_words
      else:
        probability = vocab["<unk>"][0] / num_words
      if probability != 0:
        entropy += probability * math.log(probability, 2)

    entropies.append(-entropy)
    utt_entropy.append(response_len[-1] * entropies[-1])

  # Distinct unigrams and bigrams.

  # Write to file all metrics.
  write_metric(response_len, "length")
  write_metric(entropies, "word entropy")
  write_metric(utt_entropy, "utterance entropy")
  write_metric(embedding["avg"], "embedding average")
  write_metric(embedding["extrem"], "embedding extrema")
  write_metric(embedding["greedy"], "embedding greedy")
  write_metric([len(word_set) / test_words], "distinct-1")
  write_metric([len(bigrams) / test_words], "distinct-2")

  test_responses.close()
  gt_responses.close()


# Measures cosine similarity between source and response.
def similarity(num_words, emb_dim):
  source_file = open(test_source_path)
  target_file = open(test_responses_path)
  similarities = []

  for source, target in zip(source_file, target_file):
    emb_avg = calculate_avg_embedding(source.strip("\n").split(),
                                      target.strip("\n").split(),
                                      num_words,
                                      emb_dim)
    if emb_avg is not None:
      similarities.append(emb_avg)

  write_metric(similarities, "coherence")
  source_file.close()
  target_file.close()


def main():
  num_words, emb_dim = count_words()
  metrics(num_words, emb_dim)
  similarity(num_words, emb_dim)
  output.close()


if __name__ == "__main__":
  main()
