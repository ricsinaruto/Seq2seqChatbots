# Seq2seqChatbots &middot; [![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://ctt.ac/Zgu6I)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Paper2](https://img.shields.io/badge/Paper-Survey-yellow.svg)](https://tdk.bme.hu/VIK/DownloadPaper/asdad) [![Paper1](https://img.shields.io/badge/Paper-Research-yellow.svg)](https://www.aclweb.org/anthology/P19-1567) [![Poster](https://img.shields.io/badge/Poster-Research-pink.svg)](https://ricsinaruto.github.io/website/docs/acl_poster_h.pdf) [![Code1](https://img.shields.io/badge/code-filtering-green.svg)](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering) [![Code2](https://img.shields.io/badge/code-evaluation-green.svg)](https://github.com/ricsinaruto/dialog-eval) [![notes](https://img.shields.io/badge/paper%20notes-on%20wiki-red.svg)](https://github.com/ricsinaruto/Seq2seqChatbots/wiki/Chatbot-and-Related-Research-Paper-Notes-with-Images) [![documentation](https://img.shields.io/badge/documentation-on%20wiki-red.svg)](https://github.com/ricsinaruto/Seq2seqChatbots/wiki/API-Documentation) [![blog](https://img.shields.io/badge/Blog-post-black.svg)](https://medium.com/@richardcsaky/neural-chatbots-are-dumb-65b6b40e9bd4)  
A wrapper around [tensor2tensor](https://github.com/tensorflow/tensor2tensor) to flexibly train, interact, and generate data for neural chatbots.  
The [wiki](https://github.com/ricsinaruto/Seq2seqChatbots/wiki/Chatbot-and-Related-Research-Paper-Notes-with-Images) contains my notes and summaries of over 150 recent publications related to neural dialog modeling.
#### Data filtering code for the ACL [paper](https://www.aclweb.org/anthology/P19-1567) has been moved [here](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering).  

## Features
  :floppy_disk: &nbsp; Run your own trainings or experiment with pre-trained models  
  :white_check_mark: &nbsp; 4 different dialog datasets integrated with tensor2tensor  
  :twisted_rightwards_arrows: &nbsp; Seemlessly works with any model or hyperparameter set in tensor2tensor  
  :rocket: &nbsp;&nbsp; Easily extendable base class for dialog problems



## Setup
Run setup.py which installs required packages and steps you through downloading additional data:
```
python setup.py
```
You can download all trained models used in [this](https://www.aclweb.org/anthology/P19-1567) paper from [here](https://mega.nz/#!mI0iDCTI!qhKoBiQRY3rLg3K6nxAmd4ZMNEX4utFRvSby_0q2dwU). Each training contains two checkpoints, one for the validation loss minimum and another after 150 epochs. The data and the trainings folder structure match each other exactly.
## Usage
```
python t2t_csaky/main.py --mode=train
```
The mode argument can be one of the following four: *{[generate_data](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#generate-data), [train](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#train), [decode](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#decode), experiment}*. In the *experiment* mode you can speficy what to do inside the *experiment* function of the *[run](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/utils/run.py)* file. A detailed explanation is given below, for what each mode does.

### Config
You can control the flags and parameters of each mode directly in this [file](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/config.py). For each run that you initiate this file will be copied to the appropriate directory, so you can quickly access the parameters of any run. There are some flags that you have to set for every mode (the *FLAGS* dictionary in the config file):
* **t2t_usr_dir**: Path to the directory where my code resides. You don't have to change this, unless you rename the directory.
* **data_dir**: The path to the directory where you want to generate the source and target pairs, and other data. The dataset will be downloaded one level higher from this directory into a *raw_data* folder.
* **problem**: This is the name of a registered problem that tensor2tensor needs. Detailed in the *generate_data* section below.
All paths should be from the root of the repo.

### Generate Data
This mode will download and preprocess the data and generate source and target pairs. Currently there are 6 registered problems, that you can use besides the ones given by tensor2tensor:
* *[persona_chat_chatbot](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/persona_chat_chatbot.py)*: This problem implements the [Persona-Chat](https://arxiv.org/pdf/1801.07243.pdf) dataset (without the use of personas).
* *[daily_dialog_chatbot](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/daily_dialog_chatbot.py)*: This problem implements the [DailyDialog](http://yanran.li/dailydialog.html) dataset (without the use of topics, dialog acts or emotions).
* *[opensubtitles_chatbot](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/opensubtitles_chatbot.py)*: This problem can be used to work with the [OpenSubtitles](http://opus.nlpl.eu/OpenSubtitles2018.php) dataset.
* *[cornell_chatbot_basic](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/cornell_chatbots.py)*: This problem implements the [Cornell Movie-Dialog Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
* *[cornell_chatbot_separate_names](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/cornell_chatbots.py)*: This problem uses the same Cornell corpus, however the names of the speakers and addressees of each utterance are appended, resulting in source utterances like below.
    > BIANCA_m0 what good stuff ?  CAMERON_m0
* *[character_chatbot](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/character_chatbot.py)*: This is a general character-based problem that works with any dataset. Before using this, the .txt files generated by any of the problems above have to be placed inside the data directory, and after that this problem can be used to generate tensor2tensor character-based data files.

The *PROBLEM_HPARAMS* dictionary in the config file contains problem specific parameters that you can set before generating data:
* *num_train_shards*/*num_dev_shards*: If you want the generated train or dev data to be sharded over several files.
* *vocabulary_size*: Size of the vocabulary that we want to use for the problem. Words outside this vocabulary will be replaced with the <unk> token.
* *dataset_size*: Number of utterance pairs, if we don't want to use the full dataset (defined by 0).
* *dataset_split*: Specify a train-val-test split for the problem.
* *dataset_version*: This is only relevant to the opensubtitles dataset, since there are several versions of this dataset, you can specify the year of the dataset that you want to download.
* *name_vocab_size*: This is only relevant to the cornell problem with separate names. You can set the size of the vocabulary containing only the personas.
    


### Train
This mode allows you to train a model with the specified problem and hyperparameters. The code just calls the tensor2tensor training script, so any model that is in tensor2tensor can be used. Besides these, there is also a subclassed model with small modifications:
* *[gradient_checkpointed_seq2seq](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/models/gradient_checkpointed_seq2seq.py)*: Small modification of the lstm based seq2seq model, so that own hparams can be used entirely. Before calculating the softmax the LSTM hidden units are projected to 2048 linear units as [here](https://arxiv.org/pdf/1506.05869.pdf). Finally, I tried to implement [gradient checkpointing](https://github.com/openai/gradient-checkpointing) to this model, but currently it is taken out since it didn't give good results.

There are several additional flags that you can specify for a training run in the *FLAGS* dictionary in the config file, some of which are:
* *train_dir*: Name of the directory where the training checkpoint files will be saved.
* *model*: Name of the model: either one of the above or a tensor2tensor defined model.
* *hparams*: Specify a registered hparams_set, or leave empty if you want to define hparams in the config file. In order to specify hparams for a *seq2seq* or *transformer* model, you can use the *SEQ2SEQ_HPARAMS* and *TRANSFORMER_HPARAMS* dictionaries in the config file (check it for more details).


### Decode
With this mode you can decode from the trained models. The following parameters affect the decoding (in the *FLAGS* dictionary in the config file):
* *decode_mode*: Can be *interactive*, where you can chat with the model using the command line. *file* mode allows you to specify a file with source utterances for which to generate responses, and *dataset* mode will randomly sample the validation data provided and output responses.
* *decode_dir*: Directory where you can provide file to decode from, and outputted responses will be saved here.
* *input_file_name*: Name of the file that you have to give in *file* mode (placed in the *decode_dir*).
* *output_file_name*: Name of the file, inside *decode_dir*, where output responses will be saved.
* *beam_size*: Size of the beam, when using beam search.
* *return_beams*: If False return only the top beam, otherwise return *beam_size* number of beams.

## Results & Examples
The following results are from [these](https://tdk.bme.hu/VIK/DownloadPaper/asdad) two [papers](https://arxiv.org/abs/1905.05471).

### Loss and Metrics of [Transformer](https://arxiv.org/abs/1706.03762) Trained on [Cornell](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
<a><img src="https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/wiki_images/cornell_loss.png" align="top" height="300" ></a>

<a><img src="https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/wiki_images/cornell_metrics.png" align="top" height="100" ></a>  
TRF is the Transformer model, while RT means randomly selected responses from the training set and GT means ground truth responses. For an explanation of the metrics see the [paper](https://arxiv.org/abs/1905.05471).

### Responses from [Transformer](https://arxiv.org/abs/1706.03762) and [Seq2seq](https://arxiv.org/pdf/1506.05869.pdf) Trained on [Cornell](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and [Opensubtitles](http://opus.nlpl.eu/OpenSubtitles2018.php)
S2S is a simple seq2seq model with LSTMs trained on Cornell, others are Transformer models. Opensubtitles F is pre-trained on Opensubtitles and finetuned on Cornell.
<a><img src="https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/wiki_images/tdk_table.png" align="top" height="600" ></a>

### Loss and Metrics of [Transformer](https://arxiv.org/abs/1706.03762) Trained on [DailyDialog](https://arxiv.org/abs/1710.03957)
<a><img src="https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/wiki_images/loss_dailydialog.png" align="top" height="300" ></a>

<a><img src="https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/wiki_images/daily_dialog_metrics.png" align="top" height="120" ></a>  
TRF is the Transformer model, while RT means randomly selected responses from the training set and GT means ground truth responses. For an explanation of the metrics see the [paper](https://arxiv.org/abs/1905.05471).

### Responses from [Transformer](https://arxiv.org/abs/1706.03762) Trained on [DailyDialog](https://arxiv.org/abs/1710.03957)
<a><img src="https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/wiki_images/dailydialog_examples.png" align="top" height="819" ></a>

## Contributing
##### Check the [issues](https://github.com/ricsinaruto/Seq2seqChatbots/issues) for some additions where help is appreciated. Any contributions are welcome :heart:
##### Please try to follow the code syntax style used in the repo (flake8, 2 spaces indent, 80 char lines, commenting a lot, etc.)
**New problems** can be registered by subclassing [WordChatbot](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/word_chatbot.py), or even better to subclass [CornellChatbotBasic](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/cornell_chatbots.py) or [OpensubtitleChatbot](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/opensubtitles_chatbot.py), because they implement some additional functionalities. Usually it's enough to override the *preprocess* and *create_data* functions. Check the [documentation](https://github.com/ricsinaruto/Seq2seqChatbots/wiki/API-Documentation) for more details and see [daily_dialog_chatbot](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/problems/daily_dialog_chatbot.py) for an example.  
  
**New models** and hyperparameters can be added by following the tensor2tensor [tutorial](https://github.com/tensorflow/tensor2tensor#adding-your-own-components).

## Authors
* **[Richard Csaky](ricsinaruto.github.io)** (If you need any help with running the code: ricsinaruto@hotmail.com)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/LICENSE) file for details.  
Please include a link to this repo if you use it in your work and consider citing the following paper:
```
@InProceedings{Csaky:2017,
  title = {Deep Learning Based Chatbot Models},
  author = {Csaky, Richard},
  year = {2019},
  publisher={National Scientific Students' Associations Conference},
  url ={https://tdk.bme.hu/VIK/DownloadPaper/asdad},
  note={https://tdk.bme.hu/VIK/DownloadPaper/asdad}
}
```
