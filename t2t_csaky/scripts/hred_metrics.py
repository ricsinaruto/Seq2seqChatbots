# https://github.com/julianser/hed-dlg-truncated
import math
import sys
import os
import numpy as np
from nltk.translate import bleu_score
from scipy.spatial import distance


# A helper class for entropy-based metrics.
class EntropyMetrics():
  def __init__(self, vocab, train_distro, uni_distros, bi_distros):
    self.vocab = vocab
    self.train_distro = train_distro
    self.uni_distros = uni_distros
    self.bi_distros = bi_distros

    self.metrics = {"word unigram entropy": [],
                    "word bigram entropy": [],
                    "utterance unigram entropy": [],
                    "utterance bigram entropy": [],
                    "unigram kl divergence": [],
                    "bigram kl divergence": []}

  def update_metrics(self, resp_words, gt_words):
    uni_entropy = []
    bi_entropy = []
    word_count = len(resp_words)
    for i, word in enumerate(resp_words):
      # Calculate unigram entropy.
      word = word if self.vocab.get(word) else "<unk>"
      probability = self.train_distro["uni"].get(word)
      if probability:
        uni_entropy.append(math.log(probability, 2))

      # Calculate bigram entropy.
      if i < word_count - 1:
        w = resp_words[i + 1] if self.vocab.get(resp_words[i + 1]) else "<unk>"
        probability = self.train_distro["bi"].get((word, w))
        if probability:
          bi_entropy.append(math.log(probability, 2))

    # Check if lists are empty.
    if uni_entropy:
      entropy = -sum(uni_entropy)
      self.metrics["word unigram entropy"].append(entropy / len(uni_entropy))
      self.metrics["utterance unigram entropy"].append(entropy)
    if bi_entropy:
      entropy = -sum(bi_entropy)
      self.metrics["word bigram entropy"].append(entropy / len(bi_entropy))
      self.metrics["utterance bigram entropy"].append(entropy)

    # KL-divergence
    self.calc_kl_divergence(gt_words)

  # Calculate kl divergence between between two distributions for a sentence.
  def calc_kl_divergence(self, gt_words):
    uni_div = []
    bi_div = []
    word_count = len(gt_words)

    for i, word in enumerate(gt_words):
      if self.uni_distros["model"].get(word):
        word = word if self.vocab.get(word) else "<unk>"
        uni_div.append(math.log(self.uni_distros["gt"][word] /
                                self.uni_distros["model"][word], 2))

      if i < word_count - 1:
        word2 = gt_words[i + 1] if self.vocab.get(gt_words[i + 1]) else "<unk>"
        bigram = (word, word2)
        if self.bi_distros["model"].get(bigram):
          bi_div.append(math.log(self.bi_distros["gt"][bigram] /
                                 self.bi_distros["model"][bigram], 2))

    # Exclude divide by zero errors.
    if uni_div:
      self.metrics["unigram kl divergence"].append(sum(uni_div) / len(uni_div))
    if bi_div:
      self.metrics["bigram kl divergence"].append(sum(bi_div) / len(bi_div))


# A helper class for embedding similarity metrics.
class EmbeddingMetrics():
  def __init__(self, vocab, distro, emb_dim):
    self.vocab = vocab
    self.emb_dim = emb_dim
    self.distro = distro

    self.metrics = {"embedding average": [],
                    "embedding extrema": [],
                    "embedding greedy": [],
                    "coherence": []}

  # Calculate embedding metrics.
  def update_metrics(self, source_words, resp_words, gt_words):
    avg_source = self.avg_embedding(source_words)
    avg_resp = self.avg_embedding(resp_words)
    avg_gt = self.avg_embedding(gt_words)

    # Check for zero vectors and compute cosine similarity.
    if np.count_nonzero(avg_resp):
      if np.count_nonzero(avg_source):
        self.metrics["coherence"].append(
          1 - distance.cosine(avg_source, avg_resp))
      if np.count_nonzero(avg_gt):
        self.metrics["embedding average"].append(
          1 - distance.cosine(avg_gt, avg_resp))

    # Compute extrema embedding metric.
    extrema_resp = self.extrema_embedding(resp_words)
    extrema_gt = self.extrema_embedding(gt_words)
    if np.count_nonzero(extrema_resp) and np.count_nonzero(extrema_gt):
      self.metrics["embedding extrema"].append(
        1 - distance.cosine(extrema_resp, extrema_gt))

    # Compute greedy embedding metric.
    one_side = self.greedy_embedding(gt_words, resp_words)
    other_side = self.greedy_embedding(resp_words, gt_words)

    if one_side and other_side:
      self.metrics["embedding greedy"].append((one_side + other_side) / 2)

  # Calculate the average word embedding of a sentence.
  def avg_embedding(self, words):
    vectors = []
    for word in words:
      vector = self.vocab.get(word)
      prob = self.distro.get(word)
      if vector:
        if prob:
          vectors.append(vector[0] * 0.001 / (0.001 + prob))
        else:
          vectors.append(vector[0] * 0.001 / (0.001 + 0))

    if vectors:
      return np.sum(np.array(vectors), axis=0) / len(vectors)
    else:
      return np.zeros(self.emb_dim)

  # Calculate the extrema embedding of a sentence.
  def extrema_embedding(self, words):
    vector = np.zeros(self.emb_dim)
    for word in words:
      vec = self.vocab.get(word)
      if vec:
        for i in range(self.emb_dim):
          if abs(vec[0][i]) > abs(vector[i]):
            vector[i] = vec[0][i]
    return vector

  # Calculate the greedy embedding from one side.
  def greedy_embedding(self, words1, words2):
    y_vec = np.zeros((self.emb_dim, 1))
    x_count = 0
    y_count = 0
    cos_sim = 0
    for word in words2:
      vec = self.vocab.get(word)
      if vec:
        norm = np.linalg.norm(vec[0])
        vector = vec[0] / norm if norm else vec[0]
        y_vec = np.hstack((y_vec, (vector.reshape((self.emb_dim, 1)))))
        y_count += 1

    for word in words1:
      vec = self.vocab.get(word)
      if vec:
        norm = np.linalg.norm(vec[0])
        if norm:
          cos_sim += np.max((vec[0] / norm).reshape((1, self.emb_dim)).dot(y_vec))
          x_count += 1

    if x_count > 0 and y_count > 0:
      return cos_sim / x_count


# A helper class for distinct metrics.
class DistinctMetrics():
  def __init__(self, test_distro, gt_distro):
    self.test_distro = test_distro
    self.gt_distro = gt_distro
    self.metrics = {"distinct-1": [],
                    "distinct-2": [],
                    "distinct-1 ratio": [],
                    "distinct-2 ratio": []}

  def distinct(self, distro):
    return len(distro) / sum(list(distro.values()))

  def calculate_metrics(self):
    self.metrics["distinct-1"].append(self.distinct(self.test_distro["uni"]))
    self.metrics["distinct-2"].append(self.distinct(self.test_distro["bi"]))
    self.metrics["distinct-1 ratio"].append(
      self.metrics["distinct-1"][-1] / self.distinct(self.gt_distro["uni"]))
    self.metrics["distinct-2 ratio"].append(
      self.metrics["distinct-2"][-1] / self.distinct(self.gt_distro["bi"]))


class BleuMetrics():
  def __init__(self):
    self.metrics = {"bleu-1": [], "bleu-2": [], "bleu-3": [], "bleu-4": []}
    self.smoothing = bleu_score.SmoothingFunction().method4

  def update_metrics(self, resp, gt):
    try:
      self.metrics["bleu-1"].append(
        bleu_score.sentence_bleu(gt, resp, weights=(1, 0, 0, 0), smoothing_function=self.smoothing))
      self.metrics["bleu-2"].append(
        bleu_score.sentence_bleu(gt, resp, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing))
      self.metrics["bleu-3"].append(
        bleu_score.sentence_bleu(gt, resp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing))
      self.metrics["bleu-4"].append(
        bleu_score.sentence_bleu(gt, resp, weights=(0.25, 0.25, 0.25, 0), smoothing_function=self.smoothing))
    except KeyError:
      self.metrics["bleu-1"].append(0)
      self.metrics["bleu-2"].append(0)
      self.metrics["bleu-3"].append(0)
      self.metrics["bleu-4"].append(0)


# A class to computer several metrics.
class Metrics:
  def __init__(self,
               test_responses_path="decode_dir/DailyDialog/testTarget.txt"):
    """
    Params:
      :test_responses_path: Path to the model responses on test set.
    """
    # Paths to the different data files.
    self.paths = {
      "train_source": "data_dir/Cornell/twitter/trainSource.txt",
      "gt_responses": "data_dir/Cornell/twitter/testTarget.txt",
      "test_source": "data_dir/Cornell/twitter/testSource.txt",
      "text_vocab":
        "data_dir/Cornell/twitter/vocab.chatbot.32768",
      "vector_vocab":
        "data_dir/Cornell/twitter/twitter_vocab",
      "test_responses": test_responses_path,
      "output": test_responses_path.split(".txt")[0] + "_metrics.txt"
    }
    self.vocab = {}
    # Unigram and bigram probabilities based on train, model and test data.
    self.train_distro = {"uni": {}, "bi": {}}
    self.test_distro = {"uni": {}, "bi": {}}
    self.gt_distro = {"uni": {}, "bi": {}}

    # Build the distributions.
    self.build_distributions()

    # Initialize metrics.
    self.response_len = {"length": []}
    self.entropies = EntropyMetrics(
      self.vocab, self.train_distro, self.filtered_uni, self.filtered_bi)
    self.embedding = EmbeddingMetrics(
      self.vocab, self.train_distro["uni"], self.emb_dim)
    self.distinct = DistinctMetrics(self.test_distro, self.gt_distro)
    self.bleu = BleuMetrics()

  # Count words, load vocab files and build distributions.
  def build_distributions(self):
    # Build the word vectors.
    with open(self.paths["vector_vocab"]) as file:
      for line in file:
        line_as_list = line.split()
        vector = np.array([float(num) for num in line_as_list[1:]])
        self.vocab[line_as_list[0]] = [vector]

    self.emb_dim = list(self.vocab.values())[0][0].size

    # Extend the remaining vocab.
    with open(self.paths["text_vocab"]) as file:
      for line in file:
        line = line.strip()
        if not self.vocab.get(line):
          self.vocab[line] = [np.zeros(self.emb_dim)]

    # Go through the train file and build word and bigram frequencies.
    def build_distro(distro, path):
      with open(path) as file:
        for line in file:
          words = line.split()
          word_count = len(words)
          for i, word in enumerate(words):
            word = word if self.vocab.get(word) else "<unk>"
            w_in_dict = distro["uni"].get(word)
            distro["uni"][word] = distro["uni"][word] + 1 if w_in_dict else 1

            # Bigrams.
            if i < word_count - 1:
              word2 = words[i + 1] if self.vocab.get(words[i + 1]) else "<unk>"
              bi = (word, word2)
              bigram_in_dict = distro["bi"].get(bi)
              distro["bi"][bi] = distro["bi"][bi] + 1 if bigram_in_dict else 1

    # Converts frequency dict to probabilities
    def convert_to_probs(freq_dict):
      num_words = sum(list(freq_dict.values()))
      return dict([(key, val / num_words) for key, val in freq_dict.items()])

    # Filter test and ground truth distributions, only keep intersection.
    def filter_distros(test, true):
      intersection = set.intersection(set(test.keys()), set(true.keys()))

      def probability_distro(distro):
        distro = dict(distro)
        for key in list(distro.keys()):
          if key not in intersection:
            del distro[key]
        return convert_to_probs(distro)

      test = probability_distro(test)
      true = probability_distro(true)
      return test, true

    # Build the three distributions.
    build_distro(self.train_distro, self.paths["train_source"])
    build_distro(self.test_distro, self.paths["test_responses"])
    build_distro(self.gt_distro, self.paths["gt_responses"])

    # Get probabilities for train distro.
    self.train_distro["uni"] = convert_to_probs(self.train_distro["uni"])
    self.train_distro["bi"] = convert_to_probs(self.train_distro["bi"])

    # Only keep intersection of test and ground truth distros.
    test, true = filter_distros(self.test_distro["uni"], self.gt_distro["uni"])
    self.filtered_uni = {"model": test, "gt": true}
    test, true = filter_distros(self.test_distro["bi"], self.gt_distro["bi"])
    self.filtered_bi = {"model": test, "gt": true}

  # Compute all metrics.
  def metrics(self):
    sources = open(self.paths["test_source"])
    responses = open(self.paths["test_responses"])
    gt_responses = open(self.paths["gt_responses"])

    # Loop through the test and ground truth responses, and calculate metrics.
    for source, response, target in zip(sources, responses, gt_responses):
      gt_words = target.split()
      resp_words = response.split()
      source_words = source.split()
      self.response_len["length"].append(len(resp_words))

      # Calculate metrics.
      self.entropies.update_metrics(resp_words, gt_words)
      self.embedding.update_metrics(source_words, resp_words, gt_words)
      self.bleu.update_metrics(resp_words, gt_words)
    self.distinct.calculate_metrics()

    sources.close()
    gt_responses.close()
    responses.close()

  # Compute mean, std and confidence, and write the given metric to file.
  def write_metrics(self):
    metrics = {**self.response_len,
               **self.entropies.metrics,
               **self.embedding.metrics,
               **self.distinct.metrics,
               **self.bleu.metrics}

    with open(self.paths["output"], "w") as output:
      for name, metric in metrics.items():
        length = len(metric)
        avg = sum(metric) / length
        std = np.std(metric) if length > 1 else 0

        # 95% confidence interval (t=1.97)
        confidence = 1.97 * std / math.sqrt(length)

        # Write the metric to file.
        m = name + ": " + str(avg) + " " + str(std) + " " + str(confidence)
        print(m)
        output.write(m + '\n')


def main():
  #m = Metrics(sys.argv[1]) if len(sys.argv) > 1 else Metrics()
  #m.metrics()
  #m.write_metrics()
  
  folder = "decode_dir/Cornell/twitter_all/"
  for file_name in os.listdir(folder):
    m = Metrics(folder + file_name)
    m.metrics()
    m.write_metrics()
  


if __name__ == "__main__":
  main()
