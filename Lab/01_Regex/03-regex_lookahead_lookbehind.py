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


# TASK 3:
# Use the lookaround capabilities to match the following
#    (see the lookaround slide in lecture 3 as a start)
#    a password with at least 6 characters,
#                    containing 2 upper case letters,
#                    two digits and
#                    two punctuation marks
#    the above password, but it can't start with AB, ab, 01 or 12
#       to trap the really obvious abcd... and 1234... passwords
#    separate out the different components of a camelCase word
#       (e.g. getElementById -> [get, Element, by, Id]) -
#       - Hint: identify each location where the previous character is lowercase
#                                            and the next character is uppercase
#

########################################################################################################################
# Determine which Python libraries you must import
import re

# Write your code here

################################################# Password Check #######################################################
print("\n")
passwords = [
    # Valid Passwords:
    "XyZ12!@#", "SecureP@ss77!$", "Tricky#99Aa!", "hH1!hH2!!",
    "Comp!ex99AA@", "Passw0rD!!#", "G00dP@ss##!", "Str0ng!P@ssW!",
    # Invalid Passwords:
    "ABcd12!", "abXY34!@", "0123XY!@", "12XY34!@",
    "passWord!1", "123456!", "abcdefG1!", "weakpass12",
    "NoSpecial99", "SHORT1!", "OnlyLowercase!", "ALLUPPERCASE1!",
    "1234567890!!", "Simple1!", "abcdEF12", "p@ssW1!", "12abcdXY!"]

# At least 6 characters,containing 2 upper case letters, two digits and two punctuation marks
Pattern_1 = r"^(?=(?:.*[A-Z]){2,})(?=(?:.*[0-9]){2,})(?=(?:.*\W){2,}).{6,}$"
# It can't start with AB, ab, 01 or 12
Pattern_2 = r"^(?!(?:ab|AB|01|12)).{6,}$"

Valid_Password1 = []
Valid_Password2 = []
for psw in passwords:
    match1 = re.fullmatch(Pattern_1, psw)
    if match1:
        Valid_Password1.append(match1.group())
# print(Valid_Password1)
for psw in Valid_Password1:
    match2 = re.fullmatch(Pattern_2, psw)
    if match2:
        Valid_Password2.append(match2.group())
print(f"The valid passwords are: {Valid_Password2}\n")

########################################################################################################################

############################################# Separate CamelCase Word ##################################################

CamelCase_Test = [
"getElementById", "parseHTTPRequest", "convertToJSON",
"findNextAvailableSlot", "handleUserLoginEvent", "deepLearningModel"]

# the previous character is lowercase and the next character is uppercase
Pattern_3 = r"(?=[A-Z])(?<=[a-z])|(?<=[A-Z])(?=[A-Z][a-z])"

for word in CamelCase_Test:
    match3 = re.split(Pattern_3, word)
    print(word)
    print(match3)

#############################################################################################################################
