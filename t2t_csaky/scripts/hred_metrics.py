from collections import Counter
from datasketch import MinHash
import math

train_corpus_path = "data_dir/DailyDialog/base_with_numbers/trainSource.txt"
vocab_file_path = "data_dir/DailyDialog/base_with_numbers/vocab.chatbot.16384"
test_responses_path = "decode_dir/DailyDialog/trf_20_dropout-base_target_based_identity_clustering/test_set_13k.txt"
test_source_path = "data_dir/DailyDialog/base_with_numbers/testSource.txt"
word_counts = Counter()

output = open(test_responses_path.strip(".txt") + "_metrics.txt", "w")


def count_words():
  train_corpus = open(train_corpus_path)
  vocab_file = open(vocab_file_path)
  num_words = 0

  # First load the vocab.
  for line in vocab_file:
    word_counts[line.strip("\n")] = 0
  vocab_file.close()

  # Go through the train file and build frequency Counter.
  for line in train_corpus:
    for word in line.strip("\n").split():
      num_words += 1
      if word in word_counts:
        word_counts[word] += 1
      else:
        word_counts["<unk>"] += 1

  train_corpus.close()
  vocab_file.close()
  return num_words


def average_entropy(num_words):
  test_responses = open(test_responses_path)
  entropies = []
  response_len = []

  for line in test_responses:
    words = line.strip("\n").split()
    response_len.append(len(words))

    entropy = 0
    for word in words:
      probability = word_counts[word] / num_words
      if probability != 0:
        entropy += probability * math.log(probability, 2)

    entropies.append(-entropy)

  avg_length = sum(response_len) / len(response_len)
  avg_entropy = sum(entropies) / len(entropies)

  # Compute the standard deviation.
  length_std = math.sqrt(
      sum([(x - avg_length) ** 2 for x in response_len]) /
      (len(response_len) - 1))
  entropy_std = math.sqrt(
      sum([(x - avg_entropy) ** 2 for x in entropies]) / (len(entropies) - 1))

  length = "average length: " + str(avg_length) + " (" + str(length_std) + ")"
  entropy = "average entropy: " + str(avg_entropy) + " (%f)" % (entropy_std)
  print(length)
  print(entropy)
  output.write(length + "\n")
  output.write(entropy + "\n")
  test_responses.close()


def similarity():
  source_file = open(test_source_path)
  target_file = open(test_responses_path)
  source_list = [line.strip("\n") for line in source_file]
  target_list = [line.strip("\n") for line in target_file]
  similarities = []

  for source, target in zip(source_list, target_list):
    source_hash = MinHash(num_perm=256)
    for word in source.split():
      source_hash.update(word.encode('utf8'))

    target_hash = MinHash(num_perm=256)
    for word in target.split():
      target_hash.update(word.encode('utf8'))

    similarities.append(source_hash.jaccard(target_hash))

  avg_similarity = sum(similarities) / len(similarities)
  sim_std = math.sqrt(
      sum([(x - avg_similarity) ** 2 for x in similarities]) /
      (len(similarities) - 1))

  sim = "average similarity: " + str(avg_similarity) + " (%f)" % (sim_std)
  print(sim)
  output.write(sim + "\n")

  source_file.close()
  target_file.close()


def main():
  num_words = count_words()
  average_entropy(num_words)
  similarity()
  output.close()


if __name__ == "__main__":
  main()
