"""
In this file you can set all tensor2tensor flags, hparams and any other settings
for the current run. This file will also be copied to the directory that you provide.
"""

FLAGS={
  "t2t_usr_dir"       :"t2t_csaky", # this is the directory from which tensor2tensor imports
  "data_dir"          :"data_dir/DailyDialog/base_with_numbers",
  "train_dir"         :"train_dir/DailyDialog/trf_40_dropout-base_with_numbers",
  "decode_dir"        :"decode_dir/DailyDialog/trf_20_dropout-base",
  "problem"           :"daily_dialog_chatbot",
  "model"             :"transformer",
  "hparams"           :"",  # this is empty if we use hparams defined in this file,
                            # otherwise you have to specify a registered hparams_set
  "profile_perform"   :"True",

  # training related flags
  "train_mode"        :"train_and_evaluate",
  "keep_checkpoints"  :3,       # how many checkpoints to keep behind the newest one
  "train_steps"       :1000000,
  "save_every_n_hour" :0,       # save checkpoints every n hours
  "save_every_n_secs" :1800,       # save checkpoints every n seconds, overrides the previous param
  "evaluation_steps"  :1000,    # number of evaluation steps to run at each cycle
  "evaluation_freq"   :1000,    # evaluation cycle is run every n training steps

  # decoding related flags
  "output_file_name"  :"inference_at_11k.txt",  # print the inference outputs to this
  "input_file_name"   :"NCM_examples.txt",      # read the inputs to inference from here
  "decode_mode"       :"interactive",           # can be: interactive, file, dataset
  "beam_size"         :10,
  "return_beams"      :"True"                   # if False return only the top beam, otherwise beam_size beams
}

DATA_FILTERING={
  "data_dir"          :"data_dir/DailyDialog/base_with_numbers/filtered_data/hash_jaccard/word_1000_clusters_0.9",
  "filter_problem"    :"hash_jaccard",  # can be: hash_jaccard, sentence_embedding, rnn_state
  "filter_type"       :"target_based",  # can be: target_based, source_based, both
  "num_clusters"      :5000,
  "num_permutations"  :128,    # only for hash based clustering
  "character_level"   :False,  # only for hash based clustering
  "treshold"          :0.9    # percentage treshold of entropy based filtering
}

PROBLEM_HPARAMS={
  "num_train_shards"  :1,
  "num_dev_shards"    :1,
  "vocabulary_size"   :16384,
  "dataset_size"      :0,
  "dataset_split"     :{"train":80, "val":10, "test":10},
  "dataset_version"   :2012,  # only for opensubtitles
  "name_vocab_size"   :3000   # only for cornell names problem
}

# These will be applied on top of the transformer_base hparams_set
TRANSFORMER_HPARAMS={
  # my hparams
  "roulette_wheel"    :"Normal",  # only works with roulette_transformer
  "roulette_beam_size":100,       # only works with roulette_transformer

  # hparams_set override
  "batch_size"        :4096,
  "layer_dropout"     :0.4,
  "attention_dropout" :0.2,
  "relu_dropout"      :0.2,
  "summarize_vars"    :True   # print out the model parameters at the start of training
}

# These will be applied on top of the lstm_seq2seq hparams_set
SEQ2SEQ_HPARAMS= {
  # my hparams
  "lstm_hidden_size"  :2600,

  # hparams_set override
  "optimizer"         :"Adafactor",
  "fixed_batch_size"  :False,   # if True, batch size refers to number of sentences, otherwise it's number of tokens
  "summarize_vars"    :True,
  "embed_num_shards"  :10,      # shard the embedding matrix into n different matrices
  "embedding_size"    :2048,
  "num_layers"        :2,
  "batch_size"        :512,
  "max_sentence_len"  :64,      # sentences in the dataset longer than this will be ignored
  "shared_embedding_and_softmax_weights":True   # if true, use 1 matrix for the softmax and the embedding weights
}
