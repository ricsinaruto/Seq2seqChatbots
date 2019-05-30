import os

folder = "decode_dir/Cornell/twitter_all/"
output = open(folder + "latex.txt", "w")


def round_(number):
  if len(number.split('.')[0]) > 2:
    return number.split('.')[0]
  elif len(number.split('.')[0]) > 1:
    return str(round(float(number[:6]), 1))
  elif number.split('.')[0] != '0':
    return str(round(float(number[:6]), 2))

  for i, e in enumerate(number.split(".")[1]):
    if e != '0':
      return str(round(float(number[:i + 7]), i + 3))


for file_name in os.listdir(folder):
  if "metrics" in file_name:
    with open(folder + file_name) as file:
      ckpt = file_name.split("_")[0]
      output.write(ckpt + "&")

      # Loop through metrics.
      for i, line in enumerate(file):
        if i < 11 or i > 14:
          #output.write(round_(line.split(":")[1].split()[0]) + " (" + round_(line.split(":")[1].split()[1]) + ")&")
          try:
            output.write(round_(line.split(":")[1].split()[0]) + "&")
          except TypeError:
            output.write("0.0" + "&")
        elif i < 13:
          output.write(round_(line.split(":")[1].split()[0]) + "&")
      output.write("\\\ \hline\n")
