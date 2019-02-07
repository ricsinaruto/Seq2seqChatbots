# https://github.com/julianser/hed-dlg-truncated
import math
import numpy as np
from scipy.spatial import distance


# A helper class for entropy-based metrics.
class EntropyMetrics():
  def __init__(vocab, train_distro, uni_distros, bi_distros):
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
        word2 = words[i + 1] if self.vocab.get(words[i + 1]) else "<unk>"
        probability = self.train_distro["bi"].get((word, word2))
        if probability:
          bi_entropy.append(math.log(probability, 2))

    uni_entropy = -sum(uni_entropy)
    bi_entropy = -sum(bi_entropy)
    self.metrics["word unigram entropy"].append(uni_entropy / len(uni_entropy))
    self.metrics["word bigram entropy"].append(bi_entropy / len(bi_entropy))
    self.metrics["utterance unigram entropy"].append(uni_entropy)
    self.metrics["utterance bigram entropy"].append(bi_entropy)

    # TODO
    unigram_div, bigram_div = calc_kl_divergence(gt_words)

  # Calculate kl divergence between two lines.
  def calc_kl_divergence(gt_words):
    divergence_uni = 0
    divergence_bi = 0
    num_words = 0
    num_bigrams = 0
    word_count = len(gt_words)

    for i, word in enumerate(gt_words):
      if test_distro.get(word):
        #prob_test = 1 / len(true_distro)
        divergence_uni += math.log(true_distro[word] / test_distro[word], 2)
        num_words += 1

      if i < word_count - 1:
        bigram = (word, gt_words[i + 1])
        if test_bigram_distro.get(bigram):
          #prob_test = 1 / len(true_bigram_distro)
          divergence_bi += (math.log(true_bigram_distro[bigram] /
                            test_bigram_distro[bigram], 2))
          num_bigrams += 1

    # Exclude divide by zero errors.
    num_words = num_words if num_words else 1
    num_bigrams = num_bigrams if num_bigrams else 1
    return divergence_uni / num_words, divergence_bi / num_bigrams


# A helper class for embedding similarity metrics.
class EmbeddingMetrics():
  def __init__(self):
    pass

  def update_metrics():
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
          if norm:
            vec = vec[1] / norm
            cos_sim += np.max(vec.reshape((1, emb_dim)).dot(y_vec))
            x_count += 1

      if x_count > 0 and y_count > 0:
        return cos_sim / x_count
      else:
        return None

    one_side = score(gt_words, test_words)
    other_side = score(test_words, gt_words)

    if one_side is not None and other_side is not None:
      return (one_side + other_side) / 2
    else:
      return None

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


# A helper class for distinct metrics.
class DistinctMetrics():
  def __init__(self, test_distro, gt_distro):
    self.test_distro = test_distro
    self.gt_distro = gt_distro
    self.metrics = {"distinct-1": [],
                    "distinct-2": [],
                    "distinct-1 ratio": [],
                    "distinct-2 ratio": []}

  def get_distinct(distro):
    return len(distro) / sum(list(distro.values()))

  def calculate_metrics(self):
    self.metrics["distinct-1"].append(get_distinct(self.test_distro["uni"]))
    self.metrics["distinct-2"].append(get_distinct(self.test_distro["bi"]))
    self.metrics["distinct-1 ratio"].append(
      self.metrics["distinct-1"] / get_distinct(self.gt_distro["uni"]))
    self.metrics["distinct-2 ratio"].append(
      self.metrics["distinct-2"] / get_distinct(self.gt_distro["bi"]))

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
      "train_source": "data_dir/DailyDialog/base_with_numbers/trainSource.txt",
      "gt_responses": "data_dir/DailyDialog/base_with_numbers/testTarget.txt",
      "test_source": "data_dir/DailyDialog/base_with_numbers/testSource.txt",
      "text_vocab":
        "data_dir/DailyDialog/base_with_numbers/vocab.chatbot.16384",
      "vector_vocab":
        "data_dir/DailyDialog/base_with_numbers/vocab.chatbot.16384_vector",
      "test_responses": test_responses_path,
      "output": test_responses_path.strip(".txt") + "_metrics.txt"
    }
    self.vocab = {}
    # Unigram and bigram probabilities based on train, model and test data.
    self.train_distro = {"uni": {}, "bi": {}}
    self.test_distro = {"uni": {}, "bi": {}}
    self.gt_distro = {"uni": {}, "bi": {}}

    # Build the distributions.
    self.build_distributions()

    # Initialize metrics.
    self.response_len = []
    self.entropies = EntropyMetrics(
      self.vocab, self.train_distro, self.filtered_uni, self.filtered_bi)
    self.embedding = EmbeddingMetrics(self.vocab)
    self.distinct = DistinctMetrics(self.test_distro, self.gt_distro)

  # Count words, load vocab files and build distributions.
  def build_distributions(self):
    # Build the word vectors.
    with open(self.paths["vector_vocab"]) as file:
      for line in file:
        line_as_list = line.split()
        vector = np.array([float(num) for num in line_as_list[1:]])
        self.vocab[line_as_list[0]] = vector

    self.emb_dim = len(self.vocab[self.vocab.keys()[0]][1])

    # Extend the remaining vocab.
    with open(self.paths["text_vocab"]) as file:
      for line in file:
        line = line.strip()
        if not self.vocab.get(line):
          self.vocab[line] = np.zeros(self.emb_dim)

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
              bigram_in_dict = self.train_distro["bi"].get(bi)
              distro["bi"][bi] = distro["bi"][bi] + 1 if bigram_in_dict else 1

    # Converts frequency dict to probabilities
    def convert_to_probs(freq_dict):
      num_words = sum(list(freq_dict.values()))
      return {[(key, val / num_words) for key, val in freq_dict.items()]}

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
      self.response_len.append(len(resp_words))

      # Calculate metrics.
      self.entropies.update_metrics(resp_words, gt_words)
      self.embedding.update_metrics(source_words, resp_words, gt_words)
    self.distinct.calculate_metrics()

    sources.close()
    gt_responses.close()
    responses.close()

  # Compute mean, std and confidence, and write the given metric to file.
  def write_metric():
    """
    Params:
      :metric: A list of numbers representing a metric for each response.
      :name: Name of the metric.
    """
    with open(output)
    for metric in metrics:
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


def main():
  m = Metrics()
  m.metrics()
  m.write_metrics()


if __name__ == "__main__":
  main()
