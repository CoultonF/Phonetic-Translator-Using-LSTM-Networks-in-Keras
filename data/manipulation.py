import pandas
import math
from pandas import *
fields = ['word', 'ipa']
df = pandas.read_csv('./data/non-name-words.txt', encoding='utf-16-le', usecols=fields)

max_word_len = 5
max_ipa_len = 5
data_size = 100
drop_idxs = []
for idx, row in df.iterrows():
    exclude = set('-.')
    if len(row.word) > 5 or len(row.ipa) > 5 or (any((c in exclude) for c in row.word)):
        drop_idxs.append(idx)
df = df.drop(drop_idxs)
drop_idxs = []
for idx, row in df.iterrows():
    exclude = set('-.')
    if idx%200==0:
        drop_idxs.append(idx)
df = df.loc[drop_idxs]
df.to_csv(r'./data/100-5-letter-non-name-words-dataset.txt',encoding='utf-8', header=None, index=None, mode='a',sep=',')
