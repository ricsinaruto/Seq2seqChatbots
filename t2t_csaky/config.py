"""
In this file you can set all tensor2tensor flags, hparams and other settings
for the current run. This file will also be copied to the provided directory.
"""

FLAGS={
  "t2t_usr_dir"       :"t2t_csaky", # tensor2tensor imports from this dir
  "data_dir"          :"data_dir/DailyDialog/base_with_numbers",
  "train_dir"         :"train_dir/DailyDialog/seq2seq_base-base_with_numbers",
  "decode_dir"        :"decode_dir/DailyDialog/trf_20_dropout-base",
  "problem"           :"daily_dialog_chatbot",
  "model"             :"gradient_checkpointed_seq2seq",
  "hparams"           :"",  # this is empty if we use hparams defined in this file,
                            # otherwise you have to specify a registered hparams_set
  "profile_perform"   :"True",

  # training related flags
  "train_mode"        :"train_and_evaluate",
  "memory_fraction"   :0.95,
  "keep_checkpoints"  :3,       # how many checkpoints to keep at head
  "train_steps"       :1000000,
  "save_every_n_hour" :0,       # save checkpoints every n hours
  "save_every_n_secs" :1800,    # every n seconds, overrides hour param
  "evaluation_steps"  :1000,    # number of evaluation steps at each cycle
  "evaluation_freq"   :1000,    # evaluation cycle is run every n train steps

  # decoding related flags
  "output_file_name"  :"inference_at_11k.txt",  # save the inference outputs
  "input_file_name"   :"NCM_examples.txt",      # read inputs to be fed
  "decode_mode"       :"dataset",   # can be: interactive, file, dataset
  "beam_size"         :10,
  "return_beams"      :"True"           # if False return only the top beam, 
                                        # otherwise beam_size beams
}

DATA_FILTERING={
  "data_dir"          :"data_dir/DailyDialog/base_with_numbers/filtered_data/identity_clustering/bigram_matrix",
  "filter_problem"    :"identity_clustering",  # can be: hash_jaccard, sentence_embedding, rnn_state
  "filter_type"       :"both",  # can be: target_based, source_based, both
  "source_clusters"   :100,
  "target_clusters"   :100,
  "max_length"        :64,    # max length when constructing bigram matrix, this needs to be set to 0 in order to normal filtering to run
  "min_cluster_size"  :2,     # clusters with fewer elements won't get filtered
  "num_permutations"  :128,   # only for hash based clustering
  "character_level"   :False, # only for hash based clustering
  "treshold"          :4,   # percentage treshold of entropy based filtering
  "ckpt_number"       :22001  # only for sentence embedding clustering
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
  "embed_num_shards"  :16,    # shard the embedding matrix into n matrices
  "summarize_vars"    :True   # print out the model parameters at the start
}

# These will be applied on top of the lstm_seq2seq hparams_set
SEQ2SEQ_HPARAMS= {
  # my hparams
  "lstm_hidden_size"  :3072,

  # hparams_set override
  "optimizer"         :"Adafactor",
  "fixed_batch_size"  :False, # if True, batch size is number of sentences,
                              # otherwise it's number of tokens
  "summarize_vars"    :True,
  "embed_num_shards"  :10,    # shard the embedding matrix into n matrices
  "embedding_size"    :2048,
  "num_layers"        :2,
  "batch_size"        :512,
  "max_sentence_len"  :64,    # sentences longer than this will be ignored
  "shared_embedding_and_softmax_weights":True # if True, use 1 matrix for the 
                                              # softmax and embedding weights
}
