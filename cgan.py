import pandas
from pandas import *
import string
import pprint

pp = pprint.PrettyPrinter(indent=1)
fields = ['word', 'ipa']
words = []
IPA = []
df = pandas.read_csv('non-name-words.txt', encoding='utf-16-le', usecols=fields)

for x in df.word:
    words.append(x)


for y in df.ipa:
    IPA.append(y)

print(words[100])
print(IPA[100])

def string_vectorizer(strng, alphabet=string.ascii_lowercase + ' .-'):
    vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
    return vector
one_hot = []
one_hot = string_vectorizer(words[100])
print DataFrame(one_hot)
