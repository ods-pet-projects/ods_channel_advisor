import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from scripts.prepare_posts import recommend_channels


def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    print("accuracy", accuracy)
    for score_type in ['micro', 'macro', 'weighted', 'samples']:
        print(score_type)
        f1 = f1_score(y_val, predicted, average=score_type)
        precision = average_precision_score(y_val, predicted, average=score_type)
        print("\tF1-score", f1)
        print("\tprecision", precision)


ifile_path = '../data/users_activity.csv'
df = pd.read_csv(ifile_path).dropna(subset=['text'])

vect = TfidfVectorizer()
X = vect.fit_transform(df['text'])
y = (df.loc[:, recommend_channels] > 0).astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y)
model = OneVsRestClassifier(LogisticRegression())
print('cross_val score:')
cv_score = cross_validate(model, X_train, y_train,
                          cv=5, return_estimator=True,
                          return_train_score=True,
                          scoring='f1_macro')
print(f'tfidf_logreg score for main {len(recommend_channels)} channels')
print('cv_score f1 macro')
print(cv_score['train_score'])

print(f'val score')
y_pred = cv_score['estimator'][0].predict(X_val)
print_evaluation_scores(y_val, y_pred)
