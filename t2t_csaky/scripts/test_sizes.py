

for size in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]:
  unique = set()
  unique_bi = set()
  total_bi = 0
  total = 0
  unique_test = set()
  unique_test_bi = set()
  total_test = 0
  total_test_bi = 0
  vocab = set()

  with open("data_dir/DailyDialog/base_with_numbers/vocab.chatbot.16384") as f:
    for line in f:
      vocab.add(line.strip('\n'))

  with open("data_dir/DailyDialog/base_with_numbers/testTarget.txt") as f:
    for i, line in enumerate(f):
      if i < size:
        words = line.strip('\n').split()
        total += len(words)

        for j, word in enumerate(words):
          word = word if word in vocab else "<unk>"
          unique.add(word)
          if j < len(words) - 1:
            words[j + 1] = words[j + 1] if words[j + 1] in vocab else "<unk>"
            unique_bi.add((word, words[j + 1]))
            total_bi += 1

  with open("decode_dir/DailyDialog/trf_20_dropout-base_both_identity_clustering/test_set_7k.txt") as f:
    for i, line in enumerate(f):
      if i < size:
        words = line.strip('\n').split()
        total_test += len(words)

        for j, word in enumerate(words):
          unique_test.add(word)
          if j < len(words) - 1:
            unique_test_bi.add((word, words[j + 1]))
            total_test_bi += 1

  print((len(unique_bi) / total_bi))
