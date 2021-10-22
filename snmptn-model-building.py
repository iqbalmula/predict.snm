import pandas as pd
snmptn = pd.read_csv('test streamlit sman 30.csv')

df = snmptn.copy()
target = 'lulus'
encode = ['jenjang']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'ya':0, 'tidak':1}
def target_encode(val):
    return target_mapper[val]

df['lulus'] = df['lulus'].apply(target_encode)

X = df.drop('lulus', axis=1)
y = df['lulus']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)

import pickle
pickle.dump(clf, open('snmptn_clf.pkl', 'wb'))