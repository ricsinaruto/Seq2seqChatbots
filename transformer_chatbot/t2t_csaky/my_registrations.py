from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import wsj_parsing
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_hparams
from tensor2tensor.models import transformer
from tensor2tensor.models import lstm
from tensor2tensor.utils import t2t_model

import tensorflow as tf
import re
from collections import Counter
#from nltk import tokenizer

FLAGS = tf.flags.FLAGS


# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# Data file paths 
# TODO: make these not hardcoded but either flags or downloadable
_CORNELL_TRAIN_DATASETS = ["t2t_csaky/data/movie_lines.txt","t2t_csaky/data/movie_conversations.txt"]
_CORNELL_DEV_DATASETS = ["t2t_csaky/data/movie_lines_dev.txt","t2t_csaky/data/movie_conversations_dev.txt"]
_CORNELL_TEST_DATASETS = ["t2t_csaky/data/movie_lines_test.txt","t2t_csaky/data/movie_conversations_test.txt"]

def chatbot_lstm_hparams():
	hparams=chatbot_lstm_batch_512()
	hparams.hidden_size=1800
	return hparams

@registry.register_model
class chatbot_lstm_seq2seq(t2t_model.T2TModel):
	
	def model_fn_body(self,features):
		if self._hparams.initializer == "orthogonal":
			raise ValueError("LSTM models fail with orthogonal initializer.")
		train=self._hparams.mode==tf.estimator.ModeKeys.TRAIN
		return lstm.lstm_seq2seq_internal(
			features.get("inputs"),features["targets"],chatbot_lstm_hparams(),train)



""" my own hparams for training chatbots with lstm_seq2seq_attention """
@registry.register_hparams
def chatbot_lstm_attn():
	hparams = lstm.lstm_attention()
	hparams.max_length = 256
	hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
	hparams.optimizer_adam_epsilon = 1e-9
	hparams.learning_rate_decay_scheme = "noam"
	hparams.learning_rate = 0.1
	hparams.learning_rate_warmup_steps = 4000
	hparams.initializer_gain = 1.0
	hparams.initializer = "uniform_unit_scaling"
	hparams.weight_decay = 0.0
	hparams.optimizer_adam_beta1 = 0.9
	hparams.optimizer_adam_beta2 = 0.98
	hparams.num_sampled_classes = 0
	hparams.label_smoothing = 0.1
	hparams.learning_rate_warmup_steps = 8000
	hparams.learning_rate = 0.2
	hparams.layer_preprocess_sequence = "n"
	hparams.layer_postprocess_sequence = "da"
	hparams.layer_prepostprocess_dropout = 0.1

	hparams.hidden_size=1024
	hparams.num_hidden_layers=2
	hparams.attn_vec_size=128
	hparams.batch_size=4096
	return hparams

@registry.register_hparams
def chatbot_lstm_batch_8k():
	hparams = lstm.lstm_seq2seq()
	hparams.max_length = 256
	hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
	hparams.optimizer_adam_epsilon = 1e-9
	hparams.learning_rate_decay_scheme = "noam"
	hparams.learning_rate = 0.1
	hparams.learning_rate_warmup_steps = 4000
	hparams.initializer_gain = 1.0
	hparams.initializer = "uniform_unit_scaling"
	hparams.weight_decay = 0.0
	hparams.optimizer_adam_beta1 = 0.9
	hparams.optimizer_adam_beta2 = 0.98
	hparams.num_sampled_classes = 0
	hparams.label_smoothing = 0.1
	hparams.learning_rate_warmup_steps = 8000
	hparams.learning_rate = 0.2
	hparams.layer_preprocess_sequence = "n"
	hparams.layer_postprocess_sequence = "da"
	hparams.layer_prepostprocess_dropout = 0.1

	hparams.symbol_modality_num_shards=50
	hparams.hidden_size=256
	hparams.num_hidden_layers=2
	hparams.batch_size=8192
	return hparams

@registry.register_hparams
def chatbot_lstm_batch_4k():
	hparams = chatbot_lstm_batch_8k()
	hparams.batch_size=4096
	return hparams

@registry.register_hparams
def chatbot_lstm_batch_2k():
	hparams = chatbot_lstm_batch_8k()
	hparams.batch_size=2048
	return hparams

@registry.register_hparams
def chatbot_lstm_batch_1k():
	hparams = chatbot_lstm_batch_8k()
	hparams.batch_size=1024
	return hparams

@registry.register_hparams
def chatbot_lstm_batch_512():
	hparams = chatbot_lstm_batch_8k()
	hparams.batch_size=512
	return hparams

@registry.register_hparams
def chatbot_lstm_batch_256():
	hparams = chatbot_lstm_batch_8k()
	hparams.batch_size=256
	return hparams

@registry.register_hparams
def chatbot_lstm_batch_128():
	hparams = chatbot_lstm_batch_8k()
	hparams.batch_size=128
	return hparams

""" my own hparams for the chatbot task """
@registry.register_hparams
def chatbot_cornell_new():
	"""Set of hyperparameters."""
	hparams = common_hparams.basic_params1()	# make a copy of the basic hyperparameters

	hparams.batch_size = 2048				# in tokens per batch per gpu
	hparams.batching_mantissa_bits = 2 	# controls the number of length buckets
	hparams.max_length = 256				# setting max length in a minibatch
	hparams.min_length_bucket=6			# smallest length bucket
	hparams.length_bucket_step=1.2 		# get bucket lengths by multiplying with this number

	""" Model params """
	hparams.num_hidden_layers=6			# number of hidden layers
	hparams.hidden_size=512				# hidden dimension size
	hparams.kernel_height=3
	hparams.kernel_width=1
	hparams.layer_preprocess_sequence = "n"	# use normalization before a layer
	hparams.layer_postprocess_sequence = "da"	# use dropout and residual connection after layer
	hparams.layer_prepostprocess_dropout=0.1 	# dropout rate for layer pre and postprocessing
	# moe not used in base transformer model
	hparams.moe_hidden_sizes="2048"
	hparams.moe_num_experts=64
	hparams.moe_k=2
	hparams.moe_loss_coef=1e-2

	""" Regularization """
	hparams.norm_type = "layer"			# layer normalization
	hparams.dropout=0.0					# no dropout
	hparams.clip_grad_norm=0.0			# no gradient clipping
	hparams.grad_noise_scale=0.0

	""" Training """ 
	hparams.initializer = "uniform_unit_scaling"
	hparams.initializer_gain = 1.0
	hparams.label_smoothing = 0.1
	hparams.optimize="Adam"
	hparams.optimizer_adam_epsilon = 1e-9
	hparams.optimizer_adam_beta1=0.9
	hparams.optimizer_adam_beta2=0.98
	hparams.optimizer_momentum_momentum=0.9
	hparams.weight_decay = 0.0
	hparams.weight_noise=0.0
	hparams.learning_rate_decay_scheme = "noam"
	hparams.learning_rate_warmup_steps = 16000
	hparams.learning_rate_cosine_cycle_steps=250000
	hparams.learning_rate = 0.4

	""" Decoding """
	hparams.sampling_method="argmax"
	hparams.problem_choice="adaptive"
	hparams.multiply_embedding_mode="sqrt_depth"
	hparams.num_sampled_classes = 0

	""" Input-output """
	hparams.symbol_modality_num_shards=16
	hparams.eval_drop_long_sequences=int(False)				# drop long sequences at evaluation time
	hparams.shared_embedding_and_softmax_weights = int(True)	# share the output embeddings and softmax variables
	hparams.input_modalities="default"
	hparams.target_modality="default"
	hparams.max_input_seq_length=0
	hparams.max_target_seq_length=0
	hparams.prepend_mode="none"


	""" Add new hparams """
	hparams.add_hparam("filter_size", 2048)  
	# layer-related flags
	hparams.add_hparam("num_encoder_layers", hparams.num_hidden_layers)
	hparams.add_hparam("num_decoder_layers", hparams.num_hidden_layers)
	# attention-related flags
	hparams.add_hparam("num_heads", 8)
	hparams.add_hparam("attention_key_channels", 0)
	hparams.add_hparam("attention_value_channels", 0)
	hparams.add_hparam("ffn_layer", "conv_hidden_relu")
	hparams.add_hparam("parameter_attention_key_channels", 0)
	hparams.add_hparam("parameter_attention_value_channels", 0)
	# All hyperparameters ending in "dropout" are automatically set to 0.0
	# when not in training mode.
	hparams.add_hparam("attention_dropout", 0.0)
	hparams.add_hparam("relu_dropout", 0.0)
	hparams.add_hparam("pos", "timing")  # timing, none
	hparams.add_hparam("nbr_decoder_problems", 1)
	hparams.add_hparam("proximity_bias", int(False))
	hparams.add_hparam("use_pad_remover",int(True))

	return hparams

@registry.register_hparams
def chatbot_cornell_base():
	"""Exactly replicates the base transformer model described in the paper."""
	hparams = transformer.transformer_base()
	hparams.batch_size = 4096
	hparams.learning_rate_warmup_steps=16000
	return hparams

@registry.register_hparams
def base_trf_higher_drop():
	hparams = transformer.transformer_base()
	hparams.batch_size = 2048
	hparams.layer_prepostprocess_dropout=0.2
	hparams.attention_dropout=0.1
	hparams.relu_dropout=0.1
	return hparams

@registry.register_hparams
def base_trf_40_drop():
	hparams = transformer.transformer_base()
	hparams.batch_size = 2048
	hparams.layer_prepostprocess_dropout=0.4
	hparams.attention_dropout=0.2
	hparams.relu_dropout=0.2
	return hparams

@registry.register_hparams
def base_trf_50_drop():
	hparams = transformer.transformer_base()
	hparams.batch_size = 2048
	hparams.layer_prepostprocess_dropout=0.5
	hparams.attention_dropout=0.3
	hparams.relu_dropout=0.3
	return hparams

@registry.register_hparams
def base_trf_70_drop():
	hparams = transformer.transformer_base()
	hparams.batch_size = 2048
	hparams.layer_prepostprocess_dropout=0.7
	hparams.attention_dropout=0.5
	hparams.relu_dropout=0.5
	return hparams

@registry.register_hparams
def chatbot_transformer_batch_32k():
	hparams=chatbot_cornell_base()
	hparams.batch_size=32768
	return hparams

@registry.register_hparams
def chatbot_transformer_batch_16k():
	hparams=chatbot_cornell_base()
	hparams.batch_size=16384
	return hparams

@registry.register_hparams
def chatbot_transformer_batch_8k():
	hparams=chatbot_cornell_base()
	hparams.batch_size=8192
	return hparams

@registry.register_hparams
def chatbot_transformer_batch_4k():
	hparams=chatbot_cornell_base()
	hparams.batch_size=4096
	return hparams

@registry.register_hparams
def chatbot_transformer_batch_2k():
	hparams=chatbot_cornell_base()
	hparams.batch_size=2048
	return hparams


class Chatbot(problem.Text2TextProblem):
	""" A base class for chatbot problems. """

	@property
	def is_character_level(self):
		return False

	@property
	def num_shards(self):
		return 100

	@property
	def vocab_name(self):
		return "vocab.chatbot"

	@property
	def use_subword_tokenizer(self):
		return False

def extract_ids(dialogs):
	diags=[]
	for line in dialogs:
		line=line.split(" +++$+++ ")
		line=line[3].split(",")
		i=0
		for item in line:
			line[i]=re.sub("[^A-Z0-9]","",item)
			i+=1
		diags.append(line)
	return diags

def save_lines(in_file,dialogs,src,trg,vocab_file,voc_size,tag,vocab_list=0):
	wc=Counter()
	line_dict={}
	comma_words=Counter()

	def repl(matchobj):
		return re.sub("'"," '",str(matchobj.group(0)))

	def repl_2(matchobj):
		return re.sub("'","",str(matchobj.group(0)))
	# iterate through file
	for line in in_file:
		line=line.split(" +++$+++ ")
		token=line[0]
		line=line[4].lower()

		"""
		words = line.split()
		for word in words:
			word=word.strip("'")
			if "'" in word and word not in comma_words:
				comma_words[word]=1
			if "'" in word and word in comma_words:
				comma_words[word]+=1
		"""
		# keep some special tokens
		line = re.sub("[^a-z .?!']","",line)
		line =re.sub("[.]"," . ",line)
		line =re.sub("[?]"," ? ",line)
		line =re.sub("[!]"," ! ",line)
		# take care of apostrophes
		line=re.sub("[ ]'[ ]"," ",line)
		line=re.sub(" '[a-z]",repl_2,line)
		line=re.sub("n't"," n't",line)
		line=re.sub("[^ n]'[^ t]",repl,line)

		words = line.split()
		for word in words:
			if word in wc:
				wc[word]+=1
			else:
				wc[word]=1
		line_dict[token]=line

	most_common_words=wc.most_common(voc_size-3)
	comm_words=[]
	for w,i in most_common_words:
		comm_words.append(w)
	print(len(wc))
	i=0
	for key in line_dict:
		i+=1
		if i%10000==0: print(i)
		line=line_dict[key].split()
		for word in line:
			if tag=="train":
				if word not in comm_words:
					string=" "+word+" "
					line_dict[key]=re.sub(string," <UNK> "," "+line_dict[key]+" ")
			else:
				if word not in vocab_list:
					string=" "+word+" "
					line_dict[key]=re.sub(string," <UNK> "," "+line_dict[key]+" ")

	# get the separate dialogs
	source_file = open(src, "w")
	target_file = open(trg, "w")

	for dialog in dialogs:
		i=0
		for utterance in dialog:
			if utterance != dialog[-1] and dialog[i+1]!="L211194" and dialog[i+1]!="L1045":
				source_file.write(line_dict[utterance]+'\n')
				target_file.write(line_dict[dialog[i+1]]+'\n')
			i+=1

	source_file.close()
	target_file.close()

	# print vocabulary to a file
	if tag=="train":
		voc_file=open("data_dir/"+vocab_file,"w")
		for word,i in wc.most_common(voc_size-3):
			voc_file.write(word+'\n')
		# write UNK
		voc_file.write("<UNK>"+'\n')

def token_generator(source_path, target_path, token_vocab, eos=None):
	"""Generator for sequence-to-sequence tasks that uses tokens.

	This generator assumes the files at source_path and target_path have
	the same number of lines and yields dictionaries of "inputs" and "targets"
	where inputs are token ids from the " "-split source (and target, resp.) lines
	converted to integers using the token_map.

	Args:
		source_path: path to the file with source sentences.
		target_path: path to the file with target sentences.
		token_vocab: text_encoder.TextEncoder object.
		eos: integer to append at the end of each sequence (default: None).

	Yields:
		A dictionary {"inputs": source-line, "targets": target-line} where
		the lines are integer lists converted from tokens in the file lines.
	"""
	eos_list = [] if eos is None else [eos]
	with tf.gfile.GFile(source_path, mode="r") as source_file:
		with tf.gfile.GFile(target_path, mode="r") as target_file:
			source, target = source_file.readline(), target_file.readline()
			while source and target:
				#print(source)
				#print(target)
				try:
					source_ints = token_vocab.encode(source.strip()) + eos_list
					target_ints = token_vocab.encode(target.strip()) + eos_list
					#print(source_ints)
				except KeyError:
					print(source)
					print(target)
				yield {"inputs": source_ints, "targets": target_ints}
				source, target = source_file.readline(), target_file.readline()

def preproc_data(tag,dataset,vocab_file,voc_size,vocab_list=0):
	""" Preprocess the movie data directly from txt.
		This function returns txt filenames of source and target sentences.
		dataset: 	the first element of this list is the actual movie dialogs referenced by the dialog ids
					the second element is a file contatining the dialog utterance ids
					
	"""
	dialogs=open(dataset[1])
	dialog_list=extract_ids(dialogs)

	src=tag+"Source.txt"
	trg=tag+"Target.txt"

	convos=open(dataset[0])
	save_lines(convos,dialog_list,src,trg,vocab_file,voc_size,tag,vocab_list)

# my own problem using cornell movie subtitle database
@registry.register_problem
class ChatbotCornell32k(Chatbot):
	@property
	def targeted_vocab_size(self):
		return 2**15 # 32768

	@property
	def input_space_id(self):
		return problem.SpaceID.EN_TOK

	@property
	def target_space_id(self):
		return problem.SpaceID.EN_TOK

	def generator(self,data_dir,tmp_dir,train):
		datasets = _CORNELL_TRAIN_DATASETS if train else _CORNELL_DEV_DATASETS
		tag = "train" if train else "dev"
		s_path=tag+"Source.txt"
		t_path=tag+"Target.txt"

		opensubs_train_size=1941950464
		opensubs_val_size=882973051
		cornell_names_train_size=14365422
		cornell_names_val_size=1636214
		if train and os.path.getsize(s_path)!=opensubs_train_size:
			preproc_data(tag,datasets,self.vocab_file,self.targeted_vocab_size)
		# get vocab 
		# TODO: use data_dir argument
		vocab_file=open("data_dir/"+self.vocab_file)
		vocab_list=[]
		for word in vocab_file:
			vocab_list.append(word.strip('\n'))
		vocab_file.close()
		print("Total vocabulary size of train data: ",len(vocab_list))
		if not train and os.path.getsize(s_path)!=opensubs_val_size:
			preproc_data(tag,datasets,self.vocab_file,self.targeted_vocab_size,vocab_list)
		# reserve padding and eos
		symbolizer_vocab = text_encoder.TokenTextEncoder(None,vocab_list=vocab_list,num_reserved_ids=0)
		#print(symbolizer_vocab._token_to_id)
		return token_generator(s_path,t_path,symbolizer_vocab, EOS)

@registry.register_problem
class ChatbotOpensubs100k(ChatbotCornell32k):
	@property
	def targeted_vocab_size(self):
		return 103000

	@property
	def num_dev_shards(self):
		return 100