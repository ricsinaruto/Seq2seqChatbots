from collections import Counter
from datasketch import MinHash
import math

train_corpus_path = "data_dir/DailyDialog/base/trainSource.txt"
vocab_file_path = "data_dir/DailyDialog/base/vocab.chatbot.16384"
test_responses_path = "decode_dir/DailyDialog/trf_20_dropout-base/both_source_11k.txt"
test_source_path = "decode_dir/DailyDialog/both_source.txt"
word_counts = Counter()


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
  response_word_counts = []

  for line in test_responses:
    words = line.strip("\n").split()
    response_word_counts.append(len(words))

    entropy = 0
    for word in words:
      probability = word_counts[word] / num_words
      if probability != 0:
        entropy += probability * math.log(probability, 2)

    entropies.append(-entropy)

  avg_length = sum(response_word_counts) / len(response_word_counts)
  avg_entropy = sum(entropies) / len(entropies)

  print("averge length: " + str(avg_length))
  print("average entropy: " + str(avg_entropy))
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
  print("average similarity: " + str(avg_similarity))
  source_file.close()
  target_file.close()


def main():
  num_words = count_words()
  average_entropy(num_words)
  similarity()


if __name__ == "__main__":
  main()
