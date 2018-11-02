import pandas
import tensorflow as tf
from pandas import *
import string
widx = 0
fields = ['word', 'ipa']
words = []
IPA = []
df = pandas.read_csv('name-words.txt', encoding='utf-16-le', usecols=fields)

for x in df.word:
    words.append(x)


for y in df.ipa:
    IPA.append(y)

print(words[widx])
print(IPA[widx])

def unique(list1):

    # intilize a null list
    unique_list = []

    # traverse for all elements
    for word in list1:
        print word
        # check if exists in unique_list or not
        for letter in word:
            if letter not in unique_list:
                unique_list.append(letter)

    for x in unique_list:
        print x,
    return unique_list


alpha_chars = unique(words)
print('\n')
ipa_chars = unique(IPA)
print('\n')
print len(alpha_chars)
print len(ipa_chars)


def string_vectorizer(str_input, alphabet):
    vector = [[0 if char != symbol else 1 for char in alphabet]
                  for symbol in str_input]
    return vector

print(string_vectorizer(words[widx], alpha_chars))
print(string_vectorizer(IPA[widx], ipa_chars))

X = tf.placeholder(dtype=tf.float32, shape=[None, len(alpha_chars)])
Y = tf.placeholder(dtype=tf.float32, shape=[None, len(ipa_chars)])
