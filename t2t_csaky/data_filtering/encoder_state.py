import os
import numpy as np

# My imports.
from data_filtering.semantic_clustering import SemanticClustering
from config import FLAGS
from utils.utils import read_sentences


# This class implements the encoder state clustering method.
class EncoderState(SemanticClustering):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.decode_dir = FLAGS["decode_dir"]

  # Convenience method for creating a path to the decode directory
  def _decode_data_path(self, tag, ext=''):
    return os.path.join(self.decode_dir, '{}{}'.format(tag, ext))

  def _read(self, data_tag):
    """
    Reads and creates the data for clustering, by running the pre-trained
    modified Seq2Seq model on the given data. After the sentences
    are processed by the model, the output is the reordered dataset, with
    the corresponding hidden state representations of each sequence.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.join(file_path, '..', '..')
    self.paths[data_tag] = {
        'txt': os.path.join(
            project_path, self._data_path('{}.txt'.format(data_tag))),
        'npy': os.path.join(
            project_path, self._data_path('{}.npy'.format(data_tag)))
    }

    if not os.path.exists(
            self._data_path(self.tag + data_tag + 'Original', '.txt')):
      script_path = os.path.join(
          file_path, '..', 'scripts', 'adjust_text_to_vocab.py')
      os.system('python3 {}'.format(script_path))

    if (not os.path.exists(self.paths[data_tag]['txt']) or not
            os.path.exists(self.paths[data_tag]['npy'])):
      self.generate_encoder_states(
          self._data_path(self.tag + data_tag, '.txt'),
          self._data_path('{}.txt'.format(data_tag)))

    meaning_vectors = np.load(self.paths[data_tag]['npy'])

    sentence_dict = dict(zip(read_sentences(self.paths[data_tag]['txt']),
                             zip(read_sentences(self._data_path(
                                 self.tag + data_tag + 'Original', '.txt')),
                                 meaning_vectors)))

    file = open(
        self._data_path(self.tag + data_tag, '.txt'), 'r', encoding='utf-8')

    for index, line in enumerate(file):
      self.data_points[data_tag].append(self.DataPointClass(
          sentence_dict[line.strip()][0],
          index,
          False,
          sentence_dict[line.strip()][1]))

    file.close()

  def generate_encoder_states(self, input_file_path, output_path):
    """
    Generates the encoder hidden state representations for the provided
    input file. The output will be the reordered sentences (.txt), and the
    hidden state representations (.npy) files.
    """
    # What hparams should we use.
    if FLAGS["hparams"] == "":
      hparam_string = "general_" + FLAGS["model"] + "_hparams"
    else:
      hparam_string = FLAGS["hparams"]

    decode_mode_string = ""
    # Determine the decode mode flag.
    if FLAGS["decode_mode"] == "interactive":
      decode_mode_string = " --decode_interactive"
    elif FLAGS["decode_mode"] == "file":
      decode_mode_string = (" --decode_from_file=" + input_file_path)

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..',
                               'scripts',
                               'state_extraction.py')

    gpu_memory = str(FLAGS["memory_fraction"])
    decode_hparams = ('"beam_size=' + str(FLAGS["beam_size"]) +
                      ",return_beams=" + FLAGS["return_beams"] + '"')
    os.system("python3 {} \
                --generate_data=False \
                --t2t_usr_dir=".format(script_path) + FLAGS["t2t_usr_dir"] +
              " --data_dir=" + FLAGS["data_dir"] +
              " --problem=" + FLAGS["problem"] +
              " --output_dir=" + FLAGS["train_dir"] +
              " --model=" + FLAGS["model"] +
              " --worker_gpu_memory_fraction=" + gpu_memory +
              " --hparams_set=" + hparam_string +
              " --decode_to_file=" + output_path +
              " --decode_hparams=" + decode_hparams +
              decode_mode_string)
