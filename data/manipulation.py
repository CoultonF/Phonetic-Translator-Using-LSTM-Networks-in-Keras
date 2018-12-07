import pandas
from pandas import *
fields = ['word', 'ipa']
df = pandas.read_csv('non-name-words.txt', encoding='utf-16-le', usecols=fields)

drop_idxs = []
for idx, row in df.iterrows():
    exclude = set('-.')
    if len(row.word) > 5 or len(row.ipa) > 5 or (any((c in exclude) for c in row.word)):
        drop_idxs.append(idx)

# print(drop_idxs)
df = df.drop(drop_idxs, axis=0)
df.to_csv(r'./5-letter-non-name-words-dataset.txt',encoding='utf-16-le', header=None, index=None, mode='a',sep=',')
