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
# pre-trained BERT (masked word prediction)
# API   https://huggingface.co/docs/transformers/en/model_doc/bert
#       https://huggingface.co/docs/transformers/en/main_classes/trainer
#       https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/tokenizer
#       https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset
#       https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# code  https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
# learn https://en.wikipedia.org/wiki/Logit
# data  https://huggingface.co/datasets/stanfordnlp/imdb
#       https://huggingface.co/datasets/fancyzhx/yelp_polarity
#

#
# Load a bare BERT model transformer without any specific head layer on top (just to explore it)
#

tokenizer = transformers.AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = transformers.BertModel.from_pretrained("google-bert/bert-base-uncased")

# tokenize the input sequence (sentence). we have one seq so batch size is 1.
# BERT can take single or pairs of sequences.
# single sequence: [CLS] X [SEP]
# pair of sequences: [CLS] A [SEP] B [SEP]
# for a pair the seq is concatenated and a mask (token_type_ids) is used to identify if each token is a member of the first (0) or second (1) seq
inputs = tokenizer( "Hello, my dog is cute", return_tensors="pt" )

# the tokenized input is a 1D tensor with IDs for each WordPiece (not a one-hot encoded tensor)
# BERT WordPiece tokenizer uses a byte pair encoding (BPE), where tokens (words) are to split into smaller pieces from a 30k vocabulary of word pieces
# batch size is 1 as we only have one input seq (the 1st dimension of the tensor is the batch size)
# seq size is 8 word pieces
input_shape = inputs['input_ids'].size()
print( 'input tensor size =', input_shape )

# we can decode these back to word pieces
for t in inputs['input_ids']:
    print(tokenizer.convert_ids_to_tokens(t))

# call the embedding layer manually so we can inspect the output tensor
# the embedding layer has dim of 768
batch_size, seq_length = input_shape
embedding_output = model.embeddings(
	input_ids=inputs['input_ids'],
	position_ids=None,
	token_type_ids=inputs['token_type_ids'],
	inputs_embeds=None,
	past_key_values_length=0
)
print( 'embedded input tensor size =', embedding_output.size() )
print( 'embedded input tensor =', embedding_output )

# we can print the model architecture
# look at the layers in the BERT model
print('\n\nmodel architecture (BERT without heads):')
print( model )

# encode the input
outputs = model(**inputs)

# as this is a BERT model without the head we can use it simply to encode our input
print('\n\nencoded output tensor:')
print( 'encoded output tensor size =', outputs['last_hidden_state'].size() )
print( 'encoded output tensor size =', outputs['last_hidden_state'] )


#
# Load a BERT model with two heads on top as was done during the original pretraining
# There will be a masked language modeling head (token) and a next sentence prediction head (classification).
#

model = transformers.BertForPreTraining.from_pretrained("google-bert/bert-base-uncased")

# we can print the model architecture
# look at the layers in the BERT model
print('\n\nmodel architecture (BERT with pre-training heads):')
print( model )

outputs = model(**inputs)

# we can return the token and next seq predictions
# they will not make any sense however as we do not have a mask or seq pair
prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits

print( 'masked token prediction logits =', prediction_logits )
print( 'next seq prediction logits =', seq_relationship_logits )

#
# Load a Bert Model with a single language modeling (masked word prediction) head on top
#

model = transformers.BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

# use an input with a masked token
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
print( 'input ids = ', inputs['input_ids'] )
for seq in inputs['input_ids']:
    print( 'input seq = ', tokenizer.convert_ids_to_tokens(seq) )

# run the model and get word predictions (logits for each vocab prediction candidate) for each seq token
# run with no gradients as this is model inference not training
with torch.no_grad():
	outputs = model(**inputs)
	logits = outputs.logits

print( 'logits = ', logits )

# retrieve index of [MASK] from 1st seq in batch (only seq in fact) as an int
mask_token_index = -1
for pred_index in range( len( inputs['input_ids'][0] ) ) :
	if inputs['input_ids'][0][pred_index] == tokenizer.mask_token_id :
		mask_token_index = pred_index
		break
print( 'masked token seq index (int)= ', mask_token_index )

# there will be a logit associated with each word piece (the mighest value is the most probable)
print( 'size of predictions for marked token =', logits[0, mask_token_index].size() )
print( 'predictions for masked token =', logits[0, mask_token_index] )

# print a sample of vocab candidates logits (index 999 onwards used for WordPiece tokens except for special tokens)
pred_list = [0, 100, 101, 102, 103, 999, 1000, 2000, 3000, 4000, 5000, 6000]
for pred_index in pred_list :
	str_token = tokenizer.decode(pred_index)
	print( 'candidate (id, token, logit) =', pred_index, str_token, logits[0, mask_token_index][pred_index] )

# calc the most likely prediction using argmax
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
str_token = tokenizer.decode(predicted_token_id)
print( 'most likely masked token prediction =', str_token )

#
# Load Bert Model with a next sentence prediction (classification) head on top
#

model = transformers.BertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")

# look at the layers in the model
print('\n\nmodel architecture (BERT with next sentence prediction head):')
print( model )

# use the two sentence tokenizer constructor which will add a [SEP] token between the two sentences provided
# two completely unrelated sentences
sent_first = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
sent_next = "The sky is blue due to the shorter wavelength of blue light."
inputs = tokenizer( sent_first, sent_next, return_tensors="pt" )

for seq in inputs['input_ids']:
    print( 'input seq (unrelated sents) = ', tokenizer.convert_ids_to_tokens(seq) )

# run the model and get next sent predictions (logits for next sent prediction)
# this time run model with gradiants and report loss (loss is useful during training). labels param defines the shape for computing the sequence classification/regression loss (see API docs)
outputs = model(**inputs, labels=torch.LongTensor([1]))

loss = outputs.loss
print( 'loss =', loss )

logits = outputs.logits
print( 'logits (logit true label, logit false label) =', logits )
print( 'next sent prediction =', logits[0][0] > logits[0][1] )


# two related sentences
sent_first = "In Italy they cook pizza."
sent_next = "Pizza has a cheese topping."
inputs = tokenizer( sent_first, sent_next, return_tensors="pt" )
for seq in inputs['input_ids']:
    print( 'input seq (related sents) = ', tokenizer.convert_ids_to_tokens(seq) )
outputs = model(**inputs, labels=torch.LongTensor([1]))

loss = outputs.loss
print( 'loss =', loss )

logits = outputs.logits
print( 'logits (logit true label, logit false label) =', logits )
print( 'next sent prediction =', logits[0][0] > logits[0][1] )

#
# Load Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output)
# BERT base model has been trained on yelp-polarity dataset which is 500k+ dataset of yelp reviews for binary sentiment classification (0 = negative, 1 = positive)
# https://huggingface.co/datasets/fancyzhx/yelp_polarity
#

tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = transformers.BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

# look at the layers in the model
print('\n\nmodel architecture (BERT with sequence classification/regression head):')
print( model )

# tokenize sentence
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
for seq in inputs['input_ids']:
    print( 'input seq = ', tokenizer.convert_ids_to_tokens(seq) )

# run the model and get sentiment prediction (logits for 0 negative, 1 positive)
with torch.no_grad():
	outputs = model(**inputs)
	logits = outputs.logits

predicted_class_id = logits.argmax().item()

print( 'logits (logit 0 negative, logit 1 positive) =', logits )
print( 'sentiment =', logits[0][0] < logits[0][1] )
print( 'predicted_class_id =', predicted_class_id )

#
# Lab activity
#
# (step 1) Explore this lab!
#          Play around with this lab code to understand how it works. Lookup the classes used in the huggingface API.
#          Look at the below text classifier fine-tuning tutorial to see an example of loading datasets with huggingface.
#          https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
#          https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# (step 2) Load the IMBD dataset using huggingface load_dataset() function in datasets lib.
#          Use the pre-trained yelp review model (used previously) to classify its movie reviews - do not fine-tune the model, just use the pre-trained yelp review model.
#          https://huggingface.co/datasets/stanfordnlp/imdb
#          https://huggingface.co/docs/datasets/en/loading
# (step 3) Use the huggingface Trainer class to evaluate the pretrained model on the IMDB dataset using a micro F1 metric
#          your pre-trained model should get an macro F1 score of around 0.87
#          https://huggingface.co/docs/transformers/en/main_classes/trainer
#
# (LAB SIGNOFF) none
#

# CODE SOLUTION step 3 (try for yourself before looking at this solution)

from datasets import load_dataset
import numpy as np
import sklearn
import datasets

imdb_split = load_dataset('imdb', split=['train','test'])
#imdb_split = load_dataset('imdb', split=['train[:10%]','test[:10%]'])
imdb = datasets.DatasetDict()
imdb['train'] = imdb_split[0]
imdb['test'] = imdb_split[1]
print( 'imdb dataset =', imdb )

tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = transformers.BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

def preprocess_function( data_point ) :
	return tokenizer( data_point['text'], padding=True, truncation=True, return_tensors="pt")

encoded_sample = preprocess_function(imdb['test'][:5])
print( 'encoded sample (batch of 5) =', encoded_sample )

encoded_dataset = imdb.map( preprocess_function, batched=True )

# manually calc macro F1 for two data points (both correct)
results = sklearn.metrics.f1_score( [0, 1], [0, 1], average='macro')
print('manual F1 calc =',results)

# avoid huggingfaces evaluate.load('f1') as it seems to have an error for version of libs this lab uses
# use sklearn.metrics.f1_score instead
def compute_metrics( eval_pred ) :
	logits, labels = eval_pred
	predictions = np.argmax( logits, axis=-1 )
	return { 'f1' : sklearn.metrics.f1_score( y_pred=predictions, y_true=labels, average='macro' ) }

# manually calc macro F1 for two data points (both with logits that favour the corect answer)
logits_pred = [[0.9, 0.1], [0.1, 0.9]]
gold_labels = [0, 1]
results = compute_metrics( ( logits_pred, gold_labels ) )
print('manual F1 calc (using logits) =',results)

# eval across the entire dataset
print( 'evaluating dataset' )
trainer = transformers.Trainer(
	model=model,
	args=None,
	train_dataset=None,
	eval_dataset=encoded_dataset['test'],
	compute_metrics=compute_metrics,
)
eval_data = trainer.evaluate()
print( 'Eval of pretrained model on IMBD testset =', eval_data )

