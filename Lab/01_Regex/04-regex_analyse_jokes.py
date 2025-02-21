# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2024
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Jennifer Williams
# Created Date : 2025/01/24
# Project : Teaching
# Restriction: Content for internal use at University of Southampton only
#
######################################################################


## Optional Task 4 - Analysing Lightbulb Jokes
# There were more than 10k lightbulb jokes told on Twitter in in 2015. Here are a couple of examples of the genre.
# - Q: How many thriller writers does it take to change a lightbulb? A: Two. One to screw it almost all the way in and another to give a surprising twist at the end.
# -Do you know how many folk musicians it takes to change a lightbulb? Five. One to change the lightbulb, and four to write songs about how much better the old bulb was.
#
# The standard pattern for the opening of a lightbulb joke is _How many X does it take to change a lightbulb?_
#
# The file [lightbulbs-2015.txt](../corpus/lightbulbs-2015.txt) contains one lightbulb joke tweet per line.
# Write a python code that uses regular expressions to isolates the topic X of each of the jokes and use it to produce a summary of the top 100 topics of lightbulb humour.
# - You should throw away the answers (punchlines) to the joke, just look for the topic
# - Hint: you will need to use trial and error to deal with variations in language, case and punctuation. You may want to test sets of regular expressions in an editor such as VI to allow you to see the coverage and special cases before you commit them to python code.
# - This is unmoderated Internet gathered data. Apologies for any inappropriate language that it might contain, or any examples of offensive humour. If you find anything that should be removed, please let me know.
#
# The file [lightbulbs-2020.txt](../corpus/lightbulbs-2020.txt) contains the same data for last year. Use the regular expressions you developed to generate a top 100 summary for 2020 and compare the two years. What significant changes in topics have there been between 2015 and 2020?


# Determine which Python libraries you must import
import codecs


#download the file beforehand
fname="../../corpus/comp3225/lightbulbs-2015.txt"

#make the right regular expression to isolate just the topic
pattern="what (.*) what"

lineN=0; matchN=0
for line in codecs.open(fname,"r",encoding='utf-8'):
    lineN+=1
    match=re.search(pattern, line)
    if match:
        matchN+=1
        print(match.group(1))
print("The regexp matched %s jokes out of %s" % (matchN, lineN)) 

#now go on and do the analysis
