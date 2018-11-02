import pandas
df = pandas.read_csv('data-dict.txt', encoding='utf-16-le', low_memory=False)
for index, row in df.iterrows():
    if not isinstance(row[2], (str, unicode)):
        print (row[0])
        df.drop(index, inplace=True)
df = df.drop(['nameMatch', 'name'], 1)
df.to_csv("name-words.txt", index=False, encoding='utf-16-le')
