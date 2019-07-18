from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

from t2t_csaky.problems import word_chatbot

# End-of-sentence marker
EOS = text_encoder.EOS_ID


@registry.register_problem
class CharacterChatbot(word_chatbot.WordChatbot):
  '''
  A base class for character based chatbot problems.
  '''

  @property
  def is_character_level(self):
    return True

  @property
  def targeted_vocab_size(self):
    return 0

  @property
  def targeted_dataset_size(self):
    # Character chatbot currently only supports to run on the whole dataset.
    return 0

  def generator(self, data_dir, tmp_dir, train):
    '''
    Generate the vocab and then build train and validation t2t-datagen files.
    Four .txt files have to be present in the data_dir directory:
      trainSource.txt
      trainTarget.txt
      devSource.txt
      devTarget.txt

    Params:
      :train: Whether we are in train mode or not.
    '''
    character_vocab = text_encoder.ByteTextEncoder()
    mode = 'train' if train else 'dev'
    print('t2t_csaky_log: ' + mode + ' data generation activated.')

    sourcePath = os.path.join(data_dir, mode + 'Source.txt')
    targetPath = os.path.join(data_dir, mode + 'Target.txt')

    # Try to find the txt files.
    if os.path.isfile(sourcePath) and os.path.isfile(targetPath):
      print('t2t_csaky_log: Generating ' + mode + ' files in ' + data_dir)
      return translate.character_generator(sourcePath,
                                           targetPath,
                                           character_vocab,
                                           EOS)
    else:
      print('t2t_csaky_log: ' + mode +
            ' source or target file not found, please check ' +
            'that the following files exist in your ' + data_dir +
            ' directory and rerun this program:')
      print('  trainSource.txt')
      print('  trainTarget.txt')
      print('  devSource.txt')
      print('  devTarget.txt')
