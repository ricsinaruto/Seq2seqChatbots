import argparse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--ntokens', type=int,
                      help='number of tokens to show', default=20)
  parser.add_argument('-o', '--output', type=str,
                      help='name of the output file', default='top_tokens.txt')
  parser.add_argument('-i', '--input', type=str, help='name of the input file')
 
  args = parser.parse_args()
  tokens = {}
  with open(args.input, 'r') as fin:
    for line in fin:
      for token in line.strip().split():
        tokens[token] = tokens.get(token, 0) + 1

  freqs = sorted(tokens.items(), key=lambda x: x[1], reverse=True)[:args.ntokens]
  with open(args.output, 'w') as fou:
    for freq in freqs:
      fou.write(freq[0] + '\n')

  print(freqs)


if __name__ == '__main__':
  main()

