import os
import argparse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--tokens', type=str,
                      help='file containing the tokens '
                           'to be removed from the input file')
  parser.add_argument('-o', '--output', type=str,
                      help='name of the output file', default=None)
  parser.add_argument('-i', '--input', type=str, help='name of the input file')
 
  args = parser.parse_args()
  
  out_file = args.output
  if not out_file:
    out_file = 'filtered.txt'  

  tokens = set()
  with open(args.tokens, 'r') as ftok:
    for token in ftok:
      tokens.add(token.strip())

  with open(args.input, 'r') as fin, open(out_file, 'w') as fou:
    for line in fin:
      ou_line = []
      for word in line.strip().split():
        if word not in tokens:
          ou_line.append(word)
      fou.write(' '.join(ou_line) + '\n')


if __name__ == '__main__':
  main()

