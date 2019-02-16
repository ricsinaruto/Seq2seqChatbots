import os

folder = "decode_dir/DailyDialog/trf_20_dropout-base/dev_metrics/"
output = open(folder+"all_metrics.txt", "w")

for file_name in os.listdir(folder):
  if file_name != "all_metrics.txt":
    with open(folder + file_name) as file:
      ckpt = file_name.split("_")[2]
      output.write(ckpt + " ")

      # Loop through metrics.
      for line in file:
        output.write(line.split(":")[1].split()[0] + " ")
      output.write("\n")
