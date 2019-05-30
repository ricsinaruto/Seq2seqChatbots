import random

numbers = random.sample(list(range(0, 71517)), 9318)
outfile = open("data_dir/DailyDialog/base_with_numbers/trainTarget_random.txt", "w")

with open("data_dir/DailyDialog/base_with_numbers/trainTarget.txt") as file:
  for i, line in enumerate(file):
    if i in numbers:
      outfile.write(line)

outfile.close()
