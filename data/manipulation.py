import pandas
import math
from pandas import *
fields = ['word', 'ipa']
df = pandas.read_csv('./data/non-name-words.txt', encoding='utf-16-le', usecols=fields)

max_len = 6
data_size = 8600

drop_idxs = []
for idx, row in df.iterrows():
    exclude = set('-.')
    if len(row.word) != max_len or len(row.ipa) != max_len or (any((c in exclude) for c in row.word)):
        drop_idxs.append(idx)
df = df.drop(drop_idxs)
drop_idxs = []
if len(df.index) > data_size:
    for idx, row in df.iterrows():
        exclude = set('-.')
        if idx%math.ceil(len(df.index)/data_size)==0:
            drop_idxs.append(idx)
    df = df.loc[drop_idxs]
    df.to_csv(r'./data/8600-6-letter-non-name-words-dataset.txt',encoding='utf-8', header=None, index=None, mode='a',sep=',')
print(len(df.index))
