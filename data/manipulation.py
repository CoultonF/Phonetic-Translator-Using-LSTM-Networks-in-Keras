import pandas
from pandas import *
fields = ['word', 'ipa']
df = pandas.read_csv('small-name-words-small-dataset2.txt', encoding='utf-8', usecols=fields)

drop_idxs = []
for idx, row in df.iterrows():
    exclude = set('-.')
    if len(row.word) > 9 or len(row.ipa) > 9 or idx%3==0 or (any((c in exclude) for c in row.word)):
        drop_idxs.append(idx)

# print(drop_idxs)
df = df.drop(drop_idxs, axis=0)
df.to_csv(r'./small-name-words-small-dataset3.txt',header=None, index=None, mode='a',sep=',')
