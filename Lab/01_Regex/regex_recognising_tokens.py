# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2024
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Jennifer Williams, adapted from Les Carr
# Created Date : 2025/01/24
# Project : Teaching
# Restriction: Content for internal use at University of Southampton only
#
######################################################################

# TASK 1:
# Recognise the following particular kinds of non-word lexical tokens, using your experience to define the allowable format of each token type. A whole load of data from the internet (including some HTML code) has been put into an example test file for you.
#
#    Hashtags (which can contain alphanumeric characters, underscores and hyphens). Test on the text of some tweets that you copy and paste from twitter.com.
#    UK Postcodes like SO17 1BJ or E1 8BN.
#    UK style phone numbers like 023 80591234 or 02380 590000 or +44 (0) 23 80594479 or +44 (23) 8059 5000 or mobile numbers like 07722 175921.
#    URLs http://google.com/ or https://a.b.net/c/d/e.php#fragment.
#    Email addresses like j.williams@soton.ac.uk, and decompose into username and internet domain.



import re, codecs, nltk

import codecs
fname="../../corpus/comp3225/misctext.txt"
hashtag_re="#covid"
postcode_re="SO22 4NR"
phone_re="\+44 \(23\) 8059 5000"
#etc

for line in codecs.open(fname,"r",encoding="utf-8"):
    match=re.findall(hashtag_re, line)
    if match: print(match)
    match=re.findall(postcode_re, line)
    if match: print(match)
    match=re.findall(phone_re, line)
    if match: print(match)
    #etc


# If that was too easy for you, look up the rules for the officially allowable formats of each token type in Wikipedia. Here for example, are the official UK postcode formats:
#AA9A 9AA 	WC postcode area; EC1–EC4, NW1W, SE1P, SW1 	EC1A 1BB
#A9A 9AA 	E1, N1, W1 	W1A 0AX
#A9 9AA 	B, E, G, L, M, N, S, W 	M1 1AE
#A99 9AA 	" 	B33 8TH
#AA9 9AA 	All other postcodes 	CR2 6XH
#AA99 9AA 	" 	DN55 1PT

# The rules for email adddresses are eye-watering!



# TASK 2:
#
# The zip file guardian.zip (../corpus/guardian.zip) contains the text extracted from 118 Guardian news stories from the last year about Southampton, Portsmouth and Winchester. (One story per UTF-8 file, but you might like to combine them into a single file for ease of processing.) Starting from the regexp tokenizer example in Fig 2.12 (reproduced below), extend the set of tokens recognised to capture the following types of numeric data.
#- Numbers: -12  47.2  74,832,101
#- Time: 09:17pm
#- Money: £27.8m £8bn
#- Length: 6ft 48cm
#- Phone: +44(0)2380594479
#- Age specification: 13-year-old
#- Percentage: 14.4%
#- Temperature: 28C
#- Ordinals: 48th 22nd 1st
# You can compare your results with this list of numeric tokens (../corpus/guardian-numerics.txt).
#
#What’s the biggest financial quantity that appears in these stories? What did it relate to? What are the most common numeric tokens, and why do they appear?

text='That U.S.A. poster-print costs $12.40...'

# The following pattern is reproduced from the textbook figure 2.12.
# UNFORTUNATELY, the behaviour of NLTK has changed since version 3.1,
# so that capture groups don't work any more and every set
# of grouping parentheses () has now to explicitly declare non-capturing semantics with ?:
pattern = r'''(?x)			# set flag to allow verbose regexps 
	 (?:[A-Z]\.)+			# abbreviations, e.g. U.S.A. 
	 | \w+(?:-\w+)*			# words with optional internal hyphens 
	 | \$?\d+(?:\.\d+)?%?	# currency and percentages, e.g. $12.40, 82% 
	 | \.\.\.				# ellipsis 
	 | [][.,;"'?():-_`]		# these are separate punctuation tokens; includes ], [ 
	 '''

tokens=nltk.regexp_tokenize(text, pattern)
print(tokens)
