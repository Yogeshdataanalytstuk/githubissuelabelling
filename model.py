

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
df = pd.read_csv('sample1.csv.gz')

lst=list(df['issue_body'].unique())
df['full_text'] = df['issue_title'] + "_" + df['issue_body']
tmp = df.dropna().groupby('issue_label').apply(lambda x: x.sample(frac=.20)).copy().drop(columns=['issue_label'], axis=1).reset_index()
X = tmp['full_text'].values
y = tmp['issue_label'].values
cv = StratifiedKFold(shuffle=True, random_state=RANDOM_SEED)

pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", RandomForestClassifier()),
    ]
)
model = pipeline.fit(X,y)
import joblib
joblib.dump(model,'pred.sav')
