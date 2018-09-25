# Seq2seqChatbots
 
This repository contains the code that was written for experiments described in [this](https://tdk.bme.hu/VIK/DownloadPaper/asdad) paper, and also [data filtering](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#filter-data) methods and experiments for [this](https://www.researchgate.net/publication/327594109_Making_Chatbots_Better_by_Training_on_Less_Data) paper. Own problem, hparams and model registrations are written to the [tensor2tensor](https://github.com/tensorflow/tensor2tensor) library in order to try out different datasets with the [Transformer](https://arxiv.org/abs/1706.03762) modell for training dialog agents. The folders in the repository contain the following content:
* **docs**: Latex files and pictures required to generate [this](https://tdk.bme.hu/VIK/DownloadPaper/asdad) and [this](https://www.researchgate.net/publication/327594109_Making_Chatbots_Better_by_Training_on_Less_Data).
* **t2t_csaky**: This folder contains all the source code, more detailed description can be found lower.
* **decode_dir**: Here you can find inference outputs from the various trainings that were run.
* **wiki_images**: Contains images used for the [wiki](https://github.com/ricsinaruto/Seq2seqChatbots/wiki/Chatbot-and-Related-Research-Paper-Notes-with-Images), where I write about more than 100 publications related to chatbots.

## Quick Guide
First, install all the required packages in your python environment:
```
pip install -r requirements.txt
```

### Main and Config
In order to run something, you will have to call the [main](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/main.py) file:
```
python t2t_csaky/main.py --mode=train
```
The mode flag can be one of the following four: *{[generate_data](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#generate-data), [filter data](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#filter-data), [train](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#train), [decode](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#decode)}*. Additionally an *experiment* mode can be used, where you can speficy what to do inside the *experiment* function of the *[run](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/utils/run.py)* file. A detailed explanation is given lower, for what each mode does. With version v1.1 the main and config files were introduced, for a more streamlined experience, but if you want more freedom and want to use tensor2tensor commands directly, check the v1.0_README for the old way.
#### [Config](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/config.py)
You can control the flags and parameters of each mode directly in this file. Furthermore, for each run that you initiate this file will be copied to the appropriate directory, so you can quickly access the parameters of any run. There are some flags that you have to set for every mode (the *FLAGS* dictionary in the config file):
* **t2t_usr_dir**: Path to the directory where my code resides. You don't have to change this, unless you rename the directory.
* **data_dir**: The path to the directory where you want to generate the source and target pairs, and other data. The dataset will be downloaded one level higher from this directory into a *raw_data* folder.
* **problem**: This is the name of a registered problem that tensor2tensor needs. Detailed in the *generate_data* section below.

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
 
### Filter Data
Run this mode if you want to filter a dataset based on entropy as described [here](https://www.researchgate.net/publication/327594109_Making_Chatbots_Better_by_Training_on_Less_Data). You can choose from several working clustering methods:
* *[hash_jaccard](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/hash_jaccard.py)*: Cluster sentences based on the jaccard similarity between them, using the [datasketch](https://github.com/ekzhu/datasketch) library.
* *[identity_clustering](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/identity_clustering.py)*: This is a very simple clustering method, where only sentences that are exactly the same (syntactically) fall into one cluster.
* *[average_word_embedding](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/average_word_embedding.py)*: More sophisticated method where sentences are clustered based on their average word embedding representation.
* *[encoder_state](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/identity_clustering.py)*: Cluster sentences based on their representation from a trained seq2seq model's encoder RNN final hidden state.

The *DATA_FILTERING* dictionary in the config file contains the parameters for this mode, which you will have to set. Short explanation:
* *data_dir*: Specify the directory where the new dataset will be saved.
* *filter_problem*: Specify the name of the clustering method, can be one of the above.
* *filter_type*: Whether to filter source, target, or both sides.
* *treshold*: The entropy treshold above which source-target pairs will get filtered.
* *semantic_clustering_method*: Whether to use Kmeans or Mean shift for the semantic clustering types. Mean shift looks like the superior method, where only a radius has to be given.

You can see some results of the clustering/filtering methods in the *[filtering_visualization](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/scripts/filtering_visualization.ipynb)* jupyter notebook.

### Train
This mode allows you to train a model with the specified problem and hyperparameters. Currently there are is one subclassed model with small modifications:
* *[gradient_checkpointed_seq2seq](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/models/gradient_checkpointed_seq2seq.py)*: Small modification of the lstm based seq2seq model, so that own hparams can be used entirely. Moreover, before calculating the softmax the LSTM hidden units are projected to 2048 linear units as [here](https://arxiv.org/pdf/1506.05869.pdf). Finally, I tried to implement [gradient checkpointing](https://github.com/openai/gradient-checkpointing) to this model, but currently it is taken out since it didn't give good results.

There are several additional flags that you can specify for a training run in the *FLAGS* dictionary in the config file, some of which are:
* *train_dir*: Name of the directory where the training checkpoint files will be saved.
* *model*: Name of the model: either one of the above or a tensor2tensor defined model.
* *hparams*: Specify a registered hparams_set, or leave empty if you want to define hparams in the config file. In order to specify hparams for a *seq2seq* or *transformer* model, you can use the *SEQ2SEQ_HPARAMS* and *TRANSFORMER_HPARAMS* dictionaries in the config file (check it for more details).

### Decode
With this mode you can decode from the trained models. The following parameters affect the decoding (in the *FLAGS* dictionary in the config file):
* *decode_mode*: Can be *interactive*, where you can chat with the model using the command line. *file* mode allows you to specify a file with source utterances for which to generate responses, and *dataset* mode will randomly sample the validation data provided and output responses.
* *decode_dir*: Directory where you can provide file to decode from, and outputted responses will be saved here
* *input_file_name*: Name of the file that you have to give in *file* mode (placed in the *decode_dir*).
* *output_file_name*: Name of the file, inside *decode_dir*, where output responses will be saved.
* *beam_size*: Size of the beam, when using beam search.
* *return_beams*: If False return only the top beam, otherwise return *beam_size* number of beams.


### See [this](https://anonfile.com/54YeAbf6b6/tables.pdf) for more sample response from [this](https://www.researchgate.net/publication/327594109_Making_Chatbots_Better_by_Training_on_Less_Data) paper.

### Sample responses from various trainings
S2S is a baseline seq2seq model from [this](https://arxiv.org/pdf/1506.05869.pdf) paper, Cornell is the Transformer model trained on Cornell data, Cornell S is similar, but trained with speaker-addressee annotations. OpenSubtitles is the Transformer trained with OpenSubtitles data, and OpenSubtitles F, is the previous training finetuned (further trained) on Cornell speaker annotated data.
<a><img src="https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/docs/deep_learning_based_chatbot_models/pics/general_questions.png" align="top" height="550" ></a>

##### If you require any help with running the code or if you want the files of the trained models, write to this e-mail address. (ricsinaruto@hotmail.com)
