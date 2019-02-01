original_file = open("data_dir/DailyDialog/base_with_numbers/testSource.txt")
filtered_file = open("data_dir/DailyDialog/base_with_numbers/filtered_data/avg_word_embedding/both/testSource.txt")
out_file = open("decode_dir/DailyDialog/ae_both_source.txt", "w")

filtered_set = set()
for line in filtered_file:
  filtered_set.add(line.strip())

for line in original_file:
  line = " ".join(line.strip().split())
  if line not in filtered_set:
    out_file.write(line + "\n")

original_file.close()
filtered_file.close()
out_file.close()
