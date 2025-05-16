# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2024
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2024/11/28
# Project : Teaching
# Restriction: Content for internal use at University of Southampton only
#
######################################################################

import os, sys, codecs, json, math

#
# optional GPU memory managament to get more from limited cards
# https://pytorch.org/docs/stable/notes/cuda.html#memory-management
#
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:256'
print( 'cuda PYTORCH_CUDA_ALLOC_CONF = ', os.environ['PYTORCH_CUDA_ALLOC_CONF'] )
import sys
sys.modules['peft'] = None
sys.modules['bitsandbytes'] = None
sys.modules['loralib'] = None
import torch
import transformers

#
# report GPU memory available
#

def report_gpu_mem() :
	t = torch.cuda.get_device_properties(0).total_memory / 1000000000
	r = torch.cuda.memory_reserved(0) / 1000000000
	a = torch.cuda.memory_allocated(0) / 1000000000
	f = r-a
	print( 'cuda memory (free ', math.ceil(f), '; reserved ', math.ceil(r), '; allocated ', math.ceil(a), ')' )

if torch.cuda.is_available():
	torch.cuda.empty_cache()
	report_gpu_mem()
else:
	print('cuda not available')
	sys.exit(1)

print( 'device used = cuda' )

###########################
# fine-funed BERT Text Summarization (seq2seq generation)
# API   https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder
#       https://huggingface.co/learn/nlp-course/en/chapter7/5
#       https://huggingface.co/transformers/v4.9.2/main_classes/configuration.html
# paper https://arxiv.org/abs/1907.12461
#       https://huggingface.co/blog/encoder-decoder
#       https://huggingface.co/blog/how-to-generate
# code  https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE
#       https://colab.research.google.com/drive/1Ekd5pUeCX7VOrMx94_czTkwNtLN32Uyu
#

import datasets
import numpy as np
import sklearn
import evaluate

#
# load CNN/DailyMail dataset and make a 10k sample (which will be sufficient)
#

# train_split = 'train'
# val_split = 'validation'
# test_split = 'test'

train_split = 'train[:1%]'
val_split = 'validation[:1%]'
test_split = 'test[:1%]'


#DEBUG (1% data sizes - not for actual training, just testing code runs without error)
#train_split = 'train[:1%]'
#val_split = 'validation[:1%]'
#test_split = 'test[:1%]'

### debug run
#param_logging_steps=10
#param_save_steps=5
#param_eval_steps=80
#param_warmup_steps=20
#param_save_total_limit=3
#param_max_steps=400

#### training run (9 hours Quadro RTX 6000) ####
# {'rouge1': 0.397, 'rouge2': 0.179, 'rougeL': 0.272, 'rougeLsum': 0.272}
# rouge1 >= 0.5 is considered excellent
# rouge2 >= 0.4 is considered excellent
# train for longer for better scores
# param_logging_steps=1000
# param_save_steps=500
# param_eval_steps=8000
# param_warmup_steps=2000
# param_save_total_limit=3
# param_max_steps=40000

param_max_steps=400
param_save_steps=200
param_eval_steps=400
param_logging_steps=100
param_warmup_steps = 0
param_save_total_limit=3


train_data = datasets.load_dataset( 'cnn_dailymail', '3.0.0', split=train_split )
tokenizer = transformers.BertTokenizerFast.from_pretrained( 'bert-base-uncased' )

print( 'example train_data[0] =', train_data[0] )
print( 'train_data size =', len(train_data) )

# the CNN/DailyMail articles average around 848 tokens. the summaries of them average 57 tokens.
# for input, as bert-base-cased is limited to 512 tokens the input articles need to be truncated. as news articles tend to have the important
# information this crude truncation approach is OK (but will cause some information loss).
# for output, as the CNN/DailyMail summaries are uusally under 128 tokens we can truncate without much information loss.

encoder_max_length=512
decoder_max_length=128
batch_size = 1

def process_data_to_model_inputs( batch ):
	# tokenize the inputs (article) and labels (highlight summary) truncating according to encoder and decoder input limits
	inputs = tokenizer( batch['article'], padding='max_length', truncation=True, max_length=encoder_max_length, return_tensors='pt' )
	outputs = tokenizer( batch['highlights'], padding='max_length', truncation=True, max_length=decoder_max_length, return_tensors='pt' )

	# encoder input (input text)
	batch['input_ids'] = inputs.input_ids                         # encoder gets the input (article text)
	batch['attention_mask'] = inputs.attention_mask

	# the EncoderDecoder model will shift decoder input to the right by one token and insert a [CLS] token at index 0
	# the tokenizer we are using is from BertModel for both encoder and decoder, so it inserts an unwanted [CLS] token at the start of every text block.
	# we want this for the encoder input (training input). we dont want this for the decoder input (training output).
	# this means we need to remove the [CLS] token from our tokenized decoder input, otherwise we will get input via the EncoderDecoder model for decoder of [CLS][CLS] 
	# which would confuse training of our model a lot (so it will not learn text summarization patterns properly)
	# for more details look at the code >> shift_tokens_right() within https://github.com/huggingface/transformers/blob/main/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py

	output_ids = outputs.input_ids                                  # decoder target is output (highlight summary)
	shifted_input_ids = output_ids.new_zeros( output_ids.shape )    # make an empty tensor of the same size as output
	shifted_input_ids[:, :-1] = output_ids[:, 1:].detach().clone()  # copy everything except [CLS] token at start to empty tensor, effectively removing the [CLS] token
	shifted_input_ids[:, -1] = tokenizer.pad_token_id               # add [PAD] token to the end position, which would otherwise be a zero

	# decoder input (output text)
	# the EncoderDecoder model will generate decoder_input_ids and decoder_attention_mask from the target labels automatically
	batch['labels'] = shifted_input_ids

	# make sure that the PAD token is ignored during training so set it to a special value -100 (which is outside the vocab label space)
	# as we will get a lot of pad tokens and we dont want the model rewarded for learning to pad well
	batch['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch['labels']]

	return batch

# generate seq2seq training data and convert to tensors.
# disable map caching as this will pickle code and can cause problems if code changes (e.g. debugging)
train_data_processed = train_data.map(
	process_data_to_model_inputs, 
	batched=True, 
	batch_size=batch_size, 
	remove_columns=['article', 'highlights', 'id'],
	load_from_cache_file=False
)
train_data_processed.set_format( type="torch", columns=['input_ids', 'attention_mask', 'labels'] )

# generate seq2seq val data and convert to tensors
val_data = datasets.load_dataset( 'cnn_dailymail', '3.0.0', split=val_split )

print( 'val_data size =', len(val_data) )

val_data_processed = val_data.map(
	process_data_to_model_inputs, 
	batched=True, 
	batch_size=batch_size, 
	remove_columns=['article', 'highlights', 'id'],
	load_from_cache_file=False
)
val_data_processed.set_format( type="torch", columns=['input_ids', 'attention_mask', 'labels'] )

#
# Setup ROUGE metric for evaluation
#

rouge = evaluate.load('rouge')

def compute_metrics( pred ) :
	pred_ids, labels_ids = pred
	pred_str = tokenizer.batch_decode( pred_ids, skip_special_tokens=True )

	# replace special -100 token with padding token (undoing the replacement we did in process_data_to_model_inputs())
	labels_ids[labels_ids == -100] = tokenizer.pad_token_id
	label_str = tokenizer.batch_decode( labels_ids, skip_special_tokens=True )

	# compute rouge scores
	rouge_output = rouge.compute( predictions=pred_str, references=label_str )
	return rouge_output

# manually run an example to show how ROUGE metric works
test1_str = 'This is a predicted target sentence.'
test2_str = 'Some other gold target sentence.'

pred = tokenizer( test1_str, padding='max_length', truncation=True, max_length=decoder_max_length, return_tensors='pt' )
gold = tokenizer( test2_str, padding='max_length', truncation=True, max_length=decoder_max_length, return_tensors='pt' )
print( 'pred (tensor) =', pred )
print( 'gold (tensor) =', gold )

pred_ids = pred.input_ids
gold_ids = gold.input_ids
input_batch = ( pred_ids, gold_ids )
rouge_output = compute_metrics( input_batch )
print( 'pred (batch tensors) =',input_batch[0] )
print( 'gold (batch tensors) =',input_batch[0] )
print( 'rouge report =', rouge_output )

#
# load pre-trained models
# model encoder is an instance of BertModel and decoder is one of BertLMHeadModel
#

model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained( 'bert-base-uncased', 'bert-base-uncased' )

# set special tokens needed for beam search decoder (which will not be present from preloaded encoder BERT model)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

# set model hyperparams for use with CNN/Dailymail based on existing an model (bart-large-cnn) to avoid hyperparam searching
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True # stop when num_beams sents are finished in a batch
model.config.length_penalty = 2.0
model.config.num_beams = 4

# look at the layers in the model
print('\n\nmodel architecture (BERT encoder/decoder model):')
print( model )

# print encoder/decoder config
print('\n\nmodel config (BERT encoder/decoder model):')
print( model.config )

print('\n\nmodel params (BERT encoder/decoder model) = ', model.num_parameters() )

# The Seq2SeqTrainer extends Transformer's Trainer for encoder-decoder models.
# It allows using the model.generate() function during evaluation, which is necessary to validate the performance of encoder-decoder models on most sequence-to-sequence tasks, such as summarization.

training_args = transformers.Seq2SeqTrainingArguments(
	predict_with_generate=True,
	evaluation_strategy='steps',
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	fp16=True, 
	output_dir='bert2bert_cnn_daily_mail',
	logging_steps=param_logging_steps,
	save_steps=param_save_steps,
	eval_steps=param_eval_steps,
	warmup_steps=param_warmup_steps,
	save_total_limit=param_save_total_limit,
	max_steps=param_max_steps,
)

trainer = transformers.Seq2SeqTrainer(
	model=model,
	tokenizer=tokenizer,
	args=training_args,
	compute_metrics=compute_metrics,
	train_dataset=train_data_processed,
	eval_dataset=val_data_processed,
)

train_report = trainer.train()
print( 'training report =', train_report )

val_report = trainer.evaluate()
print( 'val report =', val_report )

#
# Evaluation
# 

def generate_summary(batch):
	inputs = tokenizer( batch['article'], padding='max_length', truncation=True, max_length=encoder_max_length, return_tensors='pt' )
	gold = tokenizer( batch['highlights'], padding='max_length', truncation=True, max_length=decoder_max_length, return_tensors='pt' )

	batch['gold_ids'] = gold.input_ids.detach().clone()

	input_ids = inputs.input_ids.to('cuda')
	attention_mask = inputs.attention_mask.to('cuda')
	outputs = model.generate( input_ids, attention_mask=attention_mask, max_new_tokens=decoder_max_length )

	batch['pred_ids'] = outputs.detach().clone()

	output_str = tokenizer.batch_decode( outputs, skip_special_tokens=True )
	batch['pred_summary'] = output_str

	return batch

test_data = datasets.load_dataset( 'cnn_dailymail', '3.0.0', split=test_split )
print( 'test_data size =', len(test_data) )

results = test_data.map( generate_summary, batched=True, batch_size=batch_size, remove_columns=['article','id'], load_from_cache_file=False )
results.set_format( type="torch", columns=['gold_ids', 'pred_ids', 'pred_summary', 'highlights'] )
print( 'example test data [0] =', results[0] )

input_batch = ( results['pred_ids'], results['gold_ids'] )
rouge_output = compute_metrics( input_batch )
print( 'test report =', rouge_output )


#
# Lab activity
#
# (step 1) Explore this lab!
#          Play around with this lab code to understand how it works. Lookup the classes used in the huggingface API.
#          Try writing some code to load the saved checkpoint and do some text summarization (without training the model from scratch)
# (step 2) Consider how the model uses its cross-attention layer
#          Read the papers and blogs on this point to understand it better.
#          Notice add_cross_attention = True for the decoder but not the encoder, and look at the module layers which reflect this.
#
#          The encoder is a stack of encoder blocks.
#          An encoder block is composed of a bi-directional self-attention layer, and two feed-forward layers
#          The decoder is a stack of decoder blocks, followed by a dense layer, called LM Head.
#          A decoder block is composed of a uni-directional self-attention layer, a cross-attention layer, and two feed-forward layers.
#          The (randomly initialized) cross-attention layer in the decoder connects the (pretrained) uni-directional self-attention layer and the two feed-forward layers.
#          The cross-attention layer is trained during fine-tuning.
#
# (LAB SIGNOFF) none
#
