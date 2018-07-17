
import argparse
import os


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('v', '--vocab', type=str)
  parser.add_argument('-ti', '--targetinput', type=str)
  parser.add_argument('-to', '--targetoutput', type=str)
  parser.add_argument('-si', '--sourceinput', type=str)
  parser.add_argument('-so', '--sourceoutput', type=str)

  data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data_dir', 'base_both_rnnstate_clustering')

  args = parser.parse_args([
        '-v', data_dir + '/' + 'vocab.chatbot.16384',
        '-si', data_dir + '/' + 'fullSourceOriginal.txt',
        '-so', data_dir + '/' + 'fullSource.txt'
        '-ti', data_dir + '/' + 'fullTargetOriginal.txt'
        '-to', data_dir + '/' + 'fullTarget.txt'
  ])

  vocab = set()
  with open(args.vocab, 'r', encoding='utf-8') as v:
    for line in v:
      vocab.add(line.strip())

  with open(args.sourceinput, 'r', encoding='utf-8') as fs_in:
    with open(args.sourceoutput, 'w', encoding='utf-8') as fs_out:
      with open(args.targetinput, 'r', encoding='utf-8') as ft_in:
        with open(args.targetotput, 'w', encoding='utf-8') as ft_out:
          for line_s, line_t in zip(fs_in, ft_in):
            line_as_list_s = []
            for word in line_s.strip().split():
              if word in vocab:
                line_as_list_s.append(word)

            line_as_list_t = []
            for word in line_t.strip().split():
              if word in vocab:
                line_as_list_t.append(word)

            fs_out.write(' '.join(line_as_list_s) + '\n')
            ft_out.write(' '.join(line_as_list_t) + '\n')


if __name__ == '__main__':
    main()
