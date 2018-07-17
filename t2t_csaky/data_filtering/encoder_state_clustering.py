
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

# my imports
from semantic_clustering import SemanticClustering
from config import DATA_FILTERING, FLAGS


class EncoderState(SemanticClustering):
  """
  This class implements the encoder state clustering method.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.decode_dir = FLAGS["decode_dir"]

  def _decode_data_path(self, tag, ext=''):
    """Convenience method for creating a path to the decode directory"""
    return os.path.join(
      self.decode_dir, '{}{}'.format(tag, ext))

  def _read(self, data_tag):
    """
    Reads and creates the data for clustering, by running the pre-trained
    modified Seq2Seq model on the given data. After the sentences
    are processed by the model, the output is the reordered dataset, with
    the corresponding hidden state representations of each sequence.
    """
    project_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), '..', '..')
    self.paths[data_tag] = {
      'txt': os.path.join(
        project_path, self._decode_data_path(data_tag, '.txt')),
      'npy': os.path.join(
        project_path, self._decode_data_path(data_tag, '.npy'))
    }

    if (not os.path.exists(self.paths[data_tag]['txt']) or
          not os.path.exists(self.paths[data_tag]['npy'])):

      self.generate_encoder_states(
        self._data_path(data_tag, '.txt'),
        '{}.txt'.format(data_tag))

    meaning_vectors = np.load(self.paths[data_tag]['npy'])

    sentence_dict = dict(zip(
      self.read_sentences(self.paths[data_tag]['txt']),
      zip(self.read_sentences(self._data_path(data_tag + 'Original', '.txt')),
          meaning_vectors)))

    file = open(self._data_path(data_tag, '.txt'), 'r',
                encoding='utf-8')

    for index, line in enumerate(file):
      self.data_points[data_tag].append(self.DataPointClass(
        sentence_dict[line.strip()][0],
        index, False, sentence_dict[line.strip()][1]))

    file.close()

  def generate_encoder_states(self, input_file_path, output_file_name):
    """
    Generates the encoder hidden state representations for the provided
    input file. The output will be the reordered sentences (.txt), and the
    hidden state representations (.npy) files.
    """
    # what hparams should we use
    if FLAGS["hparams"] == "":
      hparam_string = "general_" + FLAGS["model"] + "_hparams"
    else:
      hparam_string = FLAGS["hparams"]

    decode_mode_string = ""
    # determine the decode mode flag
    if FLAGS["decode_mode"] == "interactive":
      decode_mode_string = " --decode_interactive"
    elif FLAGS["decode_mode"] == "file":
      decode_mode_string = (" --decode_from_file="
                            + input_file_path)

    script_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      '..', 'scripts', 'state_extraction.py')

    os.system("python3 {} \
                    --generate_data=False \
                    --t2t_usr_dir=".format(script_path) + FLAGS["t2t_usr_dir"]
              + " --data_dir=" + FLAGS["data_dir"]
              + " --problem=" + FLAGS["problem"]
              + " --output_dir=" + FLAGS["train_dir"]
              + " --model=" + FLAGS["model"]
              + " --worker_gpu_memory_fraction=" + str(
      FLAGS["memory_fraction"])
              + " --hparams_set=" + hparam_string
              + " --decode_to_file=" + FLAGS["decode_dir"] + "/" +
              output_file_name
              + ' --decode_hparams="beam_size=' + str(FLAGS["beam_size"])
              + ",return_beams=" + FLAGS["return_beams"] + '"'
              + decode_mode_string)
