"""
In this file you can set all tensor2tensor flags, hparams and other settings
for the current run. This file will also be copied to the provided directory.
"""


FLAGS = {
    "t2t_usr_dir": "t2t_csaky",  # Tensor2tensor imports from this dir.
    "data_dir": "data_dir/DailyDialog/no_stop_words",
    "train_dir": "train_dir/DailyDialog/seq2seq_base-base_with_numbers",
    "decode_dir": "decode_dir/DailyDialog/trf_20_dropout-base",
    "problem": "daily_dialog_chatbot",
    "model": "transformer",
    "hparams": "",  # This is empty if we use hparams defined in this file.
                    # Otherwise you have to specify a registered hparams_set.
    "profile_perform": "True",

    # Training related flags.
    "train_mode": "train_and_evaluate",
    "memory_fraction": 0.95,    # Fraction of total GPU memory to use.
    "keep_checkpoints": 3,      # How many checkpoints to keep on disk.
    "train_steps": 1000000,
    "save_every_n_hour": 0,     # Save checkpoints every n hours.
    "save_every_n_secs": 1800,  # Every n seconds, overrides hour param.
    "evaluation_steps": 1000,   # Number of evaluation steps at each cycle.
    "evaluation_freq": 1000,    # Evaluation cycle is run every n train steps.

    # Decoding related flags.
    "batch_size": 32,
    "output_file_name": "inference_at_11k.txt",  # Save the inference outputs.
    "input_file_name": "NCM_examples.txt",  # Read inputs to be fed.
    "decode_mode": "file",  # Can be: interactive, file, dataset.
    "beam_size": 10,
    "return_beams": "True"  # If False return only top beam, else beam_size.
}

DATA_FILTERING = {
    "data_dir": "data_dir/DailyDialog/no_stop_words/filtered_data/avg_word_embedding",
    "filter_problem": "avg_embedding",  # Choose several metrics for clustering.
    "filter_type": "both",  # Can be: target_based, source_based, both.
    "source_clusters": 100,
    "target_clusters": 100,
    "max_length": 0,  # Max length sentences when constructing bigram matrix.
    "min_cluster_size": 2,  # Clusters with fewer elements won't get filtered.
    "num_permutations": 128,  # Only for hash based clustering.
    "character_level": False,  # Only for hash based clustering.
    "treshold": 3,  # Entropy threshold for filtering.
    "ckpt_number": 22001,  # Only for sentence embedding clustering.
    "semantic_clustering_method": "mean_shift",  # Kmeans or mean_shift.
    "mean_shift_bw": 0.7,  # Mean shift bandwidth.
    "use_faiss": False,  # Whether to use the library for GPU based clustering.
    "max_avg_length": 15,  # Clusters with longer sentences won't get filtered.
    "max_medoid_length": 50  # Clusters with longer medoids won't get filtered.

}

PROBLEM_HPARAMS = {
    "num_train_shards": 1,
    "num_dev_shards": 1,
    "vocabulary_size": 16384,
    "dataset_size": 0,  # If zero, take the full dataset.
    "dataset_split": {"train": 80, "val": 10, "test": 10},
    "dataset_version": 2012,  # Only for opensubtitles.
    "name_vocab_size": 3000   # Only for cornell names problem.
}

# These will be applied on top of the transformer_base hparams_set.
TRANSFORMER_HPARAMS = {
    # My hparams.
    "roulette_wheel": "Normal",  # Only works with roulette_transformer.
    "roulette_beam_size": 100,  # Only works with roulette_transformer.

    # Hparams_set override.
    "batch_size": 4096,
    "layer_dropout": 0.4,
    "attention_dropout": 0.2,
    "relu_dropout": 0.2,
    "embed_num_shards": 16,  # Shard the embedding matrix into n matrices.
    "summarize_vars": True   # Print out the model parameters at the start.
}

# These will be applied on top of the lstm_seq2seq hparams_set.
SEQ2SEQ_HPARAMS = {
    # My hparams.
    "lstm_hidden_size": 3072,

    # Hparams_set override.
    "optimizer": "Adafactor",
    "fixed_batch_size": False,  # If True, batch size is number of sentences.
                                # Otherwise it's number of tokens.
    "summarize_vars": True,
    "embed_num_shards": 10,  # Shard the embedding matrix into n matrices.
    "embedding_size": 2048,
    "num_layers": 2,
    "batch_size": 512,
    "max_sentence_len": 64,  # Sentences longer than this will be ignored.
    "shared_embedding_and_softmax_weights": True  # If True, use 1 matrix for
                                                  # softmax/embedding weights.
}
