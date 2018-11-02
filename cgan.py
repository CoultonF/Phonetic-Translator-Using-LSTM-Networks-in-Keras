from __future__ import print_function
import pandas
import tensorflow as tf
from pandas import *
import string

def unique(list1):
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for word in list1:
        # check if exists in unique_list or not
        for letter in word:
            if letter not in unique_list:
                unique_list.append(letter)
    return unique_list

def string_vectorizer(str_input, alphabet):
    vector = [[0 if char != symbol else 1 for char in alphabet]
                  for symbol in str_input]
    return vector

fields = ['word', 'ipa']
train_words = []
train_ipa = []
train_df = pandas.read_csv('non-name-words.txt', encoding='utf-16-le', usecols=fields)

for x in train_df.word:
    train_words.append(x)
for y in train_df.ipa:
    train_ipa.append(y)

test_words = []
test_ipa = []
test_df = pandas.read_csv('name-words.txt', encoding='utf-16-le', usecols=fields)

for x in test_df.word:
    test_words.append(x)
for y in test_df.ipa:
    test_ipa.append(y)

train_alpha_chars = unique(train_words)
train_ipa_chars = unique(train_ipa)

print("Character Vocab")
print(*train_alpha_chars, sep=',')
print(len(train_alpha_chars))
print("IPA Vocab")
print(*train_ipa_chars, sep=',')
print(len(train_ipa_chars))

# x_train and y_train data structure
# [ [
#  [0,0,0,. . . 0,0,0],
#   . . .
#   [0,0,0,. . . 0,0,0]
# ],
# [
#  [0,0,0,. . . 0,0,0],
#  . . .
#  [0,0,0,. . . 0,0,0]
# ] ]
x_train = []
for x in range(len(train_words)):
    x_train.append(string_vectorizer(train_words[x], train_alpha_chars))
y_train = []
for y in range(len(train_ipa)):
    y_train.append(string_vectorizer(train_ipa[y], train_ipa_chars))
x_test = []
for x in range(len(test_words)):
    x_test.append(string_vectorizer(test_words[x], train_alpha_chars))
y_test = []
for y in range(len(test_ipa)):
    y_test.append(string_vectorizer(test_ipa[y], train_ipa_chars))

X = tf.placeholder(dtype=tf.float32, shape=[None, len(train_alpha_chars)])
Y = tf.placeholder(dtype=tf.float32, shape=[None, len(train_ipa_chars)])
