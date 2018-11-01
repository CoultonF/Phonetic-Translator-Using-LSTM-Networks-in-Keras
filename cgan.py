import pandas
from pandas import *
import string
widx = 0
fields = ['word', 'ipa']
words = []
IPA = []
df = pandas.read_csv('non-name-words.txt', encoding='utf-16-le', usecols=fields)

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
    for x in list1:
        # check if exists in unique_list or not
        for letter in x:
            if letter not in unique_list:
                unique_list.append(letter)

    # print list
    for x in unique_list:
        print x,
print("the unique values from 1st list is")
unique(words)

def string_vectorizer(str_input, alphabet):
    vector = [[0 if char != symbol else 1 for char in alphabet]
                  for symbol in str_input]
    return vector

print (string_vectorizer(words[widx], string.ascii_lowercase + '.-'))
print (string_vectorizer(words[widx], string.ascii_lowercase + '.-'))
X = tf.placeholder(dtype=float32, Shape=[None, 28])
# Y=
