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
# fine-tuned BERT (seq classifier)
# API   https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/bert#transformers.BertForSequenceClassification
# code  https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb
# data  https://huggingface.co/datasets/nyu-mll/glue
#       https://nyu-mll.github.io/CoLA/
#       https://huggingface.co/datasets/stanfordnlp/imdb
# learn https://en.wikipedia.org/wiki/Phi_coefficient
#

from datasets import load_dataset
import numpy as np
import sklearn
import evaluate
import datasets

#
# load data (cola dataset) from GLUE dataset on huggingface
# https://huggingface.co/datasets/nyu-mll/glue
# COLA is the Corpus of Linguistic Acceptability. Its task is to determine if a sentence is grammatically correct or not. Labels in the dataset are if a sentence is grammatically correct (1) or not (0).
# note that the testset labels are not released, so we will evaluate using the validation set
# https://nyu-mll.github.io/CoLA/
#

#####################################################################################

# cola = load_dataset('glue', 'cola')
#
# print( 'cola dataset test size = ', len(cola['train']) )
# print( 'cola dataset train size = ', len(cola['test']) )
# print( 'cola data point 0 =', cola['train'][0] )

imdb = load_dataset('imdb', split={'train': 'train[:5000]', 'test': 'test[:1000]'})
print('imdb train size =', len(imdb['train']))
print('imdb test size =', len(imdb['test']))
print('imdb data point 0 =', imdb['train'][0])

######################################################################################

#
# load metric
# COLA uses metric Matthews Correlation Coefficient as an associatiion metric for evaluation
# https://en.wikipedia.org/wiki/Phi_coefficient
#

metric = evaluate.load('glue', 'cola')

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
value_coeff = metric.compute( predictions=fake_preds, references=fake_labels )
print( 'Matthews Correlation Coefficient value (fake pred/labels) =', value_coeff )

#
# load a pre-trained model
#

###############################################################################

# tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
# model = transformers.BertForSequenceClassification.from_pretrained( 'bert-base-uncased' )

tokenizer = transformers.AutoTokenizer.from_pretrained('textattack/bert-base-uncased-yelp-polarity')
model = transformers.BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-yelp-polarity')

##################################################################################
# look at the layers in the model
print('\n\nmodel architecture (BERT with sequence classification head):')
print( model )

#
# pre-process the cola dataset to get a set of tensors ready for training
#

################################################################################

# def preprocess_function( data_point ) :
# 	return tokenizer( data_point['sentence'], padding=True, truncation=True, return_tensors="pt" )
#
# encoded_sample = preprocess_function(cola['test'][:5])
# print( 'encoded sample (batch of 5) =', encoded_sample )
#
# encoded_dataset = cola.map( preprocess_function, batched=True )
# print( 'encoded dataset =', encoded_dataset )

def preprocess_function(data_point):
	return tokenizer(data_point['text'], padding='max_length', truncation=True, max_length=256)

encoded_sample = preprocess_function(imdb['test'][:5])
print('encoded sample (batch of 5) =', encoded_sample)

encoded_dataset = imdb.map(preprocess_function, batched=True)
print('encoded dataset =', encoded_dataset)

#####################################################################################

#
# fine-tune the model uisng the cola dataset
# you can imporve results by changing the hyperparams (e.g. train with more epochs)
#

batch_size = 16
num_epochs_to_train = 5
args = transformers.TrainingArguments(
	output_dir='bert-base-uncased-finetuned-cola',
	evaluation_strategy = 'epoch',
	save_strategy = 'epoch',
	learning_rate=2e-5,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	num_train_epochs=num_epochs_to_train,
	weight_decay=0.01,
	load_best_model_at_end=True,
	metric_for_best_model='f1',
)

# def compute_metrics(eval_pred):
# 	predictions, labels = eval_pred
# 	predictions = np.argmax( predictions, axis=1 )
# 	return metric.compute( predictions=predictions, references=labels )


from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
	logits, labels = eval_pred
	preds = np.argmax(logits, axis=1)
	return {"f1": f1_score(labels, preds, average="macro")}

print('training model')
# trainer = transformers.Trainer(
# 	model=model,
# 	args=args,
# 	train_dataset=encoded_dataset['train'],
# 	eval_dataset=encoded_dataset['validation'],
# 	tokenizer=tokenizer,
# 	compute_metrics=compute_metrics
# )

trainer = transformers.Trainer(
	model=model,
	args=args,
	train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# train_report = trainer.train()
# print( 'training report =', train_report )
#
# #
# # eval
# # use the validation dataset split as this has lebels. the test dataset split has no labels (label=-1) and will cause the model to fail as label index -1 is out of range.
# # you should get a matthews correlation score of around 0.59 with the validation dataset split and 5 epochs of training
# #
#
# eval_results = trainer.evaluate()
# print( 'eval results COLA valset =', eval_results )

train_report = trainer.train()
print('training report =', train_report)
eval_results = trainer.evaluate()
print('eval results IMDB testset =', eval_results)



#
# Lab activity
#
# (step 1) Explore this lab!
#          Play around with this lab code to understand how it works. Lookup the classes used in the huggingface API.
# (step 2) Change the code to load the IMBD dataset (see bert_masked_word lab) using huggingface load_dataset() function in datasets lib.
#          Load the pre-trained yelp review model (see bert_masked_word lab).
#          https://huggingface.co/datasets/stanfordnlp/imdb
# (step 3) Fine-tune the model for the IMDB dataset and calculate the macro F1 score
#          your fine-tuned model should get an macro F1 score of around 0.92 (after 5 epochs) which outperforms the pre-trained model's F1 score of 0.87
#          https://huggingface.co/docs/transformers/en/main_classes/trainer
#
# (LAB SIGNOFF) show screen shot of the fine-tuned model evaluation results
#

