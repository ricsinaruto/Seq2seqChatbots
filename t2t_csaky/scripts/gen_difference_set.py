original_file = open("data_dir/DailyDialog/base_with_numbers/testSource.txt")
original_target = open("data_dir/DailyDialog/base_with_numbers/testTarget.txt")
filtered_file = open("data_dir/DailyDialog/base_with_numbers/filtered_data/identity_clustering/both/testSource.txt")
out_file = open("decode_dir/DailyDialog/ic_both_source.txt", "w")
out_target = open("decode_dir/DailyDialog/ic_both_target.txt", "w")

filtered_set = set()
for line in filtered_file:
  filtered_set.add(" ".join(line.split()))

for line, trg in zip(original_file, original_target):
  line = " ".join(line.split())
  if line not in filtered_set:
    out_file.write(line + "\n")
    out_target.write(trg)

original_file.close()
filtered_file.close()
out_file.close()
original_target.close()
out_target.close()
