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

# This lab covers two main issues
# - Basic Python Capabilities for Using Regular Expressions
# - How to use regular expressions for a variety of tasks

# Regular expressions are implemented in the re package, which provides a number of functions to match regexps.
import re, codecs

# List of patterns to search for

# Text to parse

text = 'This is a string with term1.'
if re.search("term[0-4]",  text):
        print('Found first term.')
if re.search("term[5-9]",  text):
        print ('Found second term.')

# Now we've seen that re.search() will take the pattern, scan the text, and then returns a Match object corresponding to the first match.
# If no pattern is found, a None is returned. Both of these are truthy, so any successful match (even a match on a null string from an ill-advised Kleene *) will evaluate to True in a Boolean context. 

match = re.search(r"\d+",  "This is the COMP3225 module's 1st lab")

print("The regexp matched '%s' between positions %s and %s" % (match.group(0), match.start(), match.end()))


# If you used capture groups in the regular expression, they will appear as arguments 1 up to 99 of the match.groups() method. 

match = re.search(r"(\d+).*(\d+)",  "This is the COMP3225 module's 1st lab")

print("The regexp matched '%s' and '%s'" % (match.group(1), match.group(2)))


# There are three variants of the search function:
#
#    re.match() is anchored at the beginning of the search string
#    re.fullmatch() is anchored at the beginning and the end of the search string
#    re.findall() return all matches
#
#You can also look for all the matches in a string with re.findall(), but it returns a list of the actual strings matched rather than a Match object.
match = re.findall(r"\d+",  "This is the COMP3225 module's 1st lab")
print("The regexp matched '%s'" % (match))


# To embed this in a file-read-and-match-print-results code fragment that works line by line, we can do the following.
# We are using the codecs package to explicitly manage the various character sets that we might encounter. 

fname="mytest.txt"

for line in codecs.open(fname,"r",encoding="utf-8"):
    match=re.findall(r"\d+", line)
    if match: print(match)


# Finally, there are four usefull flags you can use to turn on different features in Python's implementation of regular expressions.
#
#    VERBOSE. Allow inline comments and extra whitespace.
#    IGNORECASE. Do case-insensitive matches.
#    DOTALL. Allow dot (.) to match any character, including a newline. (The default behavior of dot is to match anything, except for a newline.)
#    MULTILINE. Allow anchors (^ and $) to match the beginnings and ends of lines instead of matching the beginning and end of the whole text.
#
# They can be provided as flags to the re methods
#
#    re.match("this", "This", flags=re.IGNORECASE)
#
# or as abbreviated flags
#
#    re.match("that", "That", flags=re.I)
#
# or as inline flags in the regular expression itself
#
#    re.match("(?i)those", "Those")

