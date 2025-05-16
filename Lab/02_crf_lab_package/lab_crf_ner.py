# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2025
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2025/02/04
# Project : Teaching
#
######################################################################
#
# Code derived from sklearn_crfsuite and ELI5 tutorials (both MIT license)
# https://eli5.readthedocs.io/en/latest/tutorials/sklearn_crfsuite.html
# https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
#
######################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics, eli5
from sklearn.metrics import make_scorer
from collections import Counter
import matplotlib.pyplot as plt

def task1_word2features(sent, i):

	word = sent[i][0]
	postag = sent[i][1]

	features = {
		# basic features - token and POS tag
		'word' : word,
		'postag': postag,
	}
	if i > 0:
		# features for previous word (context)
		word_prev = sent[i-1][0]
		postag_prev = sent[i-1][1]
		features.update({
			'-1:word.lower()': word_prev.lower(),
			'-1:postag': postag_prev,
		})
	else:
		features['BOS'] = True

	if i < len(sent)-1:
		# features for next word (context)
		word_next = sent[i+1][0]
		postag_next = sent[i+1][1]
		features.update({
			'+1:word.lower()': word_next.lower(),
			'+1:postag': postag_next,
		})
	else:
		features['EOS'] = True

	return features

def task2_word2features(sent, i):

	word = sent[i][0]
	postag = sent[i][1]

	features = {
		'word' : word,
		'postag': postag,

		# token shape
		'word.lower()': word.lower(),
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),

		# token suffix
		'word.suffix': word.lower()[-3:],

		# POS prefix
		'postag[:2]': postag[:2],
	}
	if i > 0:
		word_prev = sent[i-1][0]
		postag_prev = sent[i-1][1]
		features.update({
			'-1:word.lower()': word_prev.lower(),
			'-1:postag': postag_prev,
			'-1:word.lower()': word_prev.lower(),
			'-1:word.isupper()': word_prev.isupper(),
			'-1:word.istitle()': word_prev.istitle(),
			'-1:word.isdigit()': word_prev.isdigit(),
			'-1:word.suffix': word_prev.lower()[-3:],
			'-1:postag[:2]': postag_prev[:2],
		})
	else:
		features['BOS'] = True

	if i < len(sent)-1:
		word_next = sent[i+1][0]
		postag_next = sent[i+1][1]
		features.update({
			'+1:word.lower()': word_next.lower(),
			'+1:postag': postag_next,
			'+1:word.lower()': word_next.lower(),
			'+1:word.isupper()': word_next.isupper(),
			'+1:word.istitle()': word_next.istitle(),
			'+1:word.isdigit()': word_next.isdigit(),
			'+1:word.suffix': word_next.lower()[-3:],
			'+1:postag[:2]': postag_next[:2],
		})
	else:
		features['EOS'] = True

	return features

def sent2features(sent, word2features_func = None):
	return [word2features_func(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token, postag, label in sent]

def sent2tokens(sent):
	return [token for token, postag, label in sent]

def print_F1_scores( micro_F1 ) :
	for label in micro_F1 :
		print( "%-15s -> f1 %0.2f ; prec %0.2f ; recall %0.2f" % ( label, micro_F1[label]['f1-score'], micro_F1[label]['precision'], micro_F1[label]['recall'] ) )

def print_transitions(trans_features):
	for (label_from, label_to), weight in trans_features:
		print( "%-15s -> %-15s %0.6f" % (label_from, label_to, weight) )

def print_state_features(state_features):
	for (attr, label), weight in state_features:
		print( "%0.6f %-15s %s" % (weight, label, attr) )

# Function to load a parsed JSON formatted file with the ontonotes 5.0 dataset.
# The dataset is parsed and training and testset created, each a list of sentences constisting of lists of (token, POS_tag, NER_IOB_tag)
# tuples. IOB tagging is a scheme defining #Begin, Inside, Outside tags for labels.
#
# For example "I like New York in the spring" might be tagged "O O B-LOC I-LOC O O O" for the named entity "New York".
#
# Ontonotes is an annotated dataset created from various genres of text (news, conversational telephone speech, weblogs, usenet
# newsgroups, broadcast, talk shows) in three languages (English, Chinese, and Arabic). Annotations include structural information
# (syntax and predicate argument structure) and shallow semantics (word sense linked to an ontology and coreference).
# We will only use a parsed version here with the words, POS tags and NER tags.
def create_dataset( max_files = None ) :
	# load parsed ontonotes dataset
	readHandle = codecs.open( 'ontonotes_parsed.json', 'r', 'utf-8', errors = 'replace' )
	str_json = readHandle.read()
	readHandle.close()
	dict_ontonotes = json.loads( str_json )

	# make a training and test split
	list_files = list( dict_ontonotes.keys() )
	if len(list_files) > max_files :
		list_files = list_files[ :max_files ]
	nSplit = math.floor( len(list_files)*0.9 )
	list_train_files = list_files[ : nSplit ]
	list_test_files = list_files[ nSplit : ]

	# sent = (tokens, pos, IOB_label)
	list_train = []
	for str_file in list_train_files :
		for str_sent_index in dict_ontonotes[str_file] :
			# ignore sents with non-PENN POS tags
			if 'XX' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue
			if 'VERB' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue

			list_entry = []

			# compute IOB tags for named entities (if any)
			ne_type_last = None
			for nTokenIndex in range(len(dict_ontonotes[str_file][str_sent_index]['tokens'])) :
				strToken = dict_ontonotes[str_file][str_sent_index]['tokens'][nTokenIndex]
				strPOS = dict_ontonotes[str_file][str_sent_index]['pos'][nTokenIndex]
				ne_type = None
				if 'ne' in dict_ontonotes[str_file][str_sent_index] :
					dict_ne = dict_ontonotes[str_file][str_sent_index]['ne']
					if not 'parse_error' in dict_ne :
						for str_NEIndex in dict_ne :
							if nTokenIndex in dict_ne[str_NEIndex]['tokens'] :
								ne_type = dict_ne[str_NEIndex]['type']
								break
				if ne_type != None :
					if ne_type == ne_type_last :
						strIOB = 'I-' + ne_type
					else :
						strIOB = 'B-' + ne_type
				else :
					strIOB = 'O'
				ne_type_last = ne_type

				list_entry.append( ( strToken, strPOS, strIOB ) )

			list_train.append( list_entry )

	list_test = []
	for str_file in list_test_files :
		for str_sent_index in dict_ontonotes[str_file] :
			# ignore sents with non-PENN POS tags
			if 'XX' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue
			if 'VERB' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue

			list_entry = []

			# compute IOB tags for named entities (if any)
			ne_type_last = None
			for nTokenIndex in range(len(dict_ontonotes[str_file][str_sent_index]['tokens'])) :
				strToken = dict_ontonotes[str_file][str_sent_index]['tokens'][nTokenIndex]
				strPOS = dict_ontonotes[str_file][str_sent_index]['pos'][nTokenIndex]
				ne_type = None
				if 'ne' in dict_ontonotes[str_file][str_sent_index] :
					dict_ne = dict_ontonotes[str_file][str_sent_index]['ne']
					if not 'parse_error' in dict_ne :
						for str_NEIndex in dict_ne :
							if nTokenIndex in dict_ne[str_NEIndex]['tokens'] :
								ne_type = dict_ne[str_NEIndex]['type']
								break
				if ne_type != None :
					if ne_type == ne_type_last :
						strIOB = 'I-' + ne_type
					else :
						strIOB = 'B-' + ne_type
				else :
					strIOB = 'O'
				ne_type_last = ne_type

				list_entry.append( ( strToken, strPOS, strIOB ) )

			list_test.append( list_entry )

	return list_train, list_test

def task1_train_crf_model( X_train, Y_train, max_iter, labels ) :
	# train the basic CRF model
	crf = sklearn_crfsuite.CRF(
		algorithm='lbfgs',
		c1=0.1,
		c2=0.1,
		max_iterations=max_iter,
		all_possible_transitions=False,
	)
	crf.fit(X_train, Y_train)
	return crf

def task3_train_crf_model( X_train, Y_train, max_iter, labels ) :
	# train CRF model using L1 reg of 200 (high value)
	crf = sklearn_crfsuite.CRF(
		algorithm='lbfgs',
		c1=200,
		c2=0.1,
		max_iterations=max_iter,
		all_possible_transitions=False,
	)
	crf.fit(X_train, Y_train)
	return crf

def task4_train_crf_model( X_train, Y_train, max_iter, labels ) :
	# train CRF model using all possible transitions
	crf = sklearn_crfsuite.CRF(
		algorithm='lbfgs',
		c1=0.1,
		c2=0.1,
		max_iterations=max_iter,
		all_possible_transitions=True,
	)
	crf.fit(X_train, Y_train)
	return crf

def task5_train_crf_model( X_train, Y_train, max_iter, labels ) :
	# randomized search to discover best parameters for CRF model
	crf = sklearn_crfsuite.CRF(
		algorithm='lbfgs', 
		max_iterations=max_iter, 
		all_possible_transitions=True
	)
	params_space = {
		'c1': scipy.stats.expon(scale=0.5),
		'c2': scipy.stats.expon(scale=0.05),
	}

	# optimize for micro F1 score
	f1_scorer = make_scorer( sklearn_crfsuite.metrics.flat_f1_score, average='weighted', labels=labels )

	print( 'starting randomized search for hyperparameters' )
	n_folds = 3
	n_candidates = 50
	rs = sklearn.model_selection.RandomizedSearchCV(crf, params_space, cv=n_folds, verbose=1, n_jobs=-1, n_iter=n_candidates, scoring=f1_scorer)
	rs.fit(X_train, Y_train)

	# output the results
	print( 'best params: {}'.format( rs.best_params_ ) )
	print( 'best micro F1 score: {}'.format( rs.best_score_ ) )
	print( 'model size: {:0.2f}M'.format( rs.best_estimator_.size_ / 1000000 ) )
	print( 'cv_results_ = ' + repr(rs.cv_results_) )

	# visualize the results in hyperparameter space (optional)
	_x = [s['c1'] for s in rs.cv_results_['params']]
	_y = [s['c2'] for s in rs.cv_results_['params']]
	_c = [s for s in rs.cv_results_['mean_test_score']]

	fig = plt.figure()
	fig.set_size_inches(12, 12)
	ax = plt.gca()
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlabel('C1')
	ax.set_ylabel('C2')
	ax.set_title("Randomized Hyperparameter Search - F1 scores (blue min={:0.2}, red max={:0.2})".format( min(_c), max(_c) ))
	ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])

	screen_y = 1920
	screen_x = 1080
	plt.gcf().set_size_inches( 0.8*screen_x/96, 0.8*screen_y/96 )
	plt.gcf().set_dpi( 96 )
	plt.show()

	# return the best model
	crf = rs.best_estimator_
	return crf

def exec_task( max_files = 10, max_iter = 20, display_label_subset = [], word2features_func = None, train_crf_model_func = None ) :
	print( 'max iterations = ' + repr(max_iter) )
	print( 'word2features_func = ' + word2features_func.__name__  )
	print( 'train_crf_model_func = ' + train_crf_model_func.__name__  )

	# make a dataset from english NE labelled ontonotes sents
	train_sents, test_sents = create_dataset( max_files = max_files )
	print( '# training sents = ' + str(len(train_sents)) )
	print( '# test sents = ' + str(len(test_sents)) )

	# print example sent (1st sent)
	print( '' )
	print( 'Example training sent annotated with IOB tags  = ' + repr(train_sents[0]) )

	# create feature vectors for every sent
	X_train = [sent2features(s, word2features_func = word2features_func) for s in train_sents]
	Y_train = [sent2labels(s) for s in train_sents]

	X_test = [sent2features(s, word2features_func = word2features_func) for s in test_sents]
	Y_test = [sent2labels(s) for s in test_sents]

	# get the label set
	set_labels = set([])
	for data in [Y_train,Y_test] :
		for n_sent in range(len(data)) :
			for str_label in data[n_sent] :
				set_labels.add( str_label )
	labels = list( set_labels )
	print( '' )
	print( 'labels = ' + repr(labels) )

	# remove 'O' label as we are not usually interested in how well 'O' is predicted
	#labels = list( crf.classes_ )
	labels.remove('O')

	# print example feature vector (12th word of 1st sent)
	print( '' )
	print( 'Example training feature = ' + repr(X_train[0][10]) )

	# Train CRF model
	crf = train_crf_model_func( X_train, Y_train, max_iter, labels )

	print('Top 10 features per-target (for a subset of labels)')
	print(
		eli5.format_as_text(
			eli5.explain_weights(crf, top=10, targets = display_label_subset ),
			show=['targets'] )
		)

	print('Label transition weights learnt from dataset (for a subset of labels)')
	print(
		eli5.format_as_text(
			eli5.explain_weights(crf, targets = display_label_subset ),
			show=['transition_features'] )
		)

	# compute the macro F1 score (F1 for instances of each label class averaged) in the test set
	Y_pred = crf.predict( X_test )
	sorted_labels = sorted(
		labels, 
		key=lambda name: (name[1:], name[0])
	)
	macro_scores = sklearn_crfsuite.metrics.flat_classification_report( Y_test, Y_pred, labels=sorted_labels, digits=3, output_dict = True )
	print( '' )
	print( 'macro F1 scores'  )
	print_F1_scores( macro_scores )

	# inspect the transitions
	print( '' )
	print("Top 10 likely state transitions")
	print_transitions( Counter(crf.transition_features_).most_common(10) )

	print( '' )
	print("Top 10 unlikely state transitions")
	print_transitions( Counter(crf.transition_features_).most_common()[-10:] )

	# inspect the states
	print( '' )
	print("Top 10 positive states")
	print_state_features(Counter(crf.state_features_).most_common(10))

	print( '' )
	print("Top 10 negative states")
	print_state_features(Counter(crf.state_features_).most_common()[-10:])

if __name__ == '__main__':
	task = sys.argv[1]
	print( 'task = ' + repr(task) )

	# settings (max_iter can be 150 for a better result)
	max_iter = 20
	max_files = 50
	display_label_subset = [ 'B-DATE', 'I-DATE', 'B-GPE', 'I-GPE', 'B-PERSON', 'I-PERSON', 'O' ]

	if task == 'task1' :
		# Task 1
		exec_task( word2features_func = task1_word2features, train_crf_model_func = task1_train_crf_model, max_files = max_files, max_iter = max_iter, display_label_subset = display_label_subset )

	elif task == 'task2' :
		# Task 2
		exec_task( word2features_func = task2_word2features, train_crf_model_func = task1_train_crf_model, max_files = max_files, max_iter = max_iter, display_label_subset = display_label_subset )

	elif task == 'task3' :
		# Task 3
		exec_task( word2features_func = task2_word2features, train_crf_model_func = task3_train_crf_model, max_files = max_files, max_iter = max_iter, display_label_subset = display_label_subset )

	elif task == 'task4' :
		# Task 4
		exec_task( word2features_func = task2_word2features, train_crf_model_func = task4_train_crf_model, max_files = max_files, max_iter = max_iter, display_label_subset = display_label_subset )

	elif task == 'task5' :
		# Task 5
		exec_task( word2features_func = task2_word2features, train_crf_model_func = task5_train_crf_model, max_files = max_files, max_iter = max_iter, display_label_subset = display_label_subset )

