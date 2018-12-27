import pandas
import math
from pandas import *
fields = ['word', 'ipa']
df = pandas.read_csv('./data/name-words.txt', encoding='utf-16-le', usecols=fields)
#
# for max_len in range(1,13):
#
#     drop_idxs = []
#     for idx, row in df.iterrows():
#         exclude = set('-.')
#         if len(row.ipa) != max_len or (any((c in exclude) for c in row.word)):
#             drop_idxs.append(idx)
#     out_df = df.drop(drop_idxs)
    # print(str(len(out_df.index)),',',str(max_len))
    # out_df.to_csv(r'./data/' + str(len(out_df.index)) + '-' + str(max_len) + '-letter-name-words-dataset.txt',encoding='utf-8', header=None, index=None, mode='a',sep=',')
df.to_csv(r'./data/name-words-utf8.txt',encoding='utf-8', header=None, index=None, mode='a',sep=',')
