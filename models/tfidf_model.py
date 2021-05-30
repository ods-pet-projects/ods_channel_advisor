import os

from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier

from models.base_model import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from scripts.utils import loadit, saveit


class TfidfModel(BaseModel):
    def __init__(self, model_name, cache_path):
        self.model_name = model_name
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)
        self.ofile_cv_model_scores = f'{cache_path}/cv_scores'
        self.ofile_model_path_vect = f'{cache_path}/vect'
        self.ofile_model_path = f'{cache_path}/model'
        self.model = None
        self.cv_scores = None
        self.vect = None

    def fit(self, X_text: list, y: list):
        if os.path.exists(self.ofile_model_path_vect):
            self.vect = loadit(self.ofile_model_path_vect)
        else:
            if self.model_name.startswith('tfidf'):
                self.vect = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            elif self.model_name.startswith('tfidf_ch'):
                self.vect = TfidfVectorizer(
                            sublinear_tf=True,
                            strip_accents='unicode',
                            analyzer='char',
                            stop_words='english',
                            ngram_range=(1, 5),
                            max_features=5,
                            lowercase=True)
            elif self.model_name.startswith('count'):
                self.vect = CountVectorizer(max_features=10000)
            else:
                raise Exception(f'unknown vectorizer {self.model_name}')

            self.vect.fit(X_text)
            saveit(self.vect, self.ofile_model_path_vect)

        if os.path.exists(self.ofile_model_path):
            self.model = loadit(self.ofile_model_path)
            self.cv_scores = loadit(self.ofile_cv_model_scores)
        else:
            X = self.vect.transform(X_text)
            if self.model_name.endswith('logreg'):
                basic_cls = LogisticRegression()
            elif self.model_name.endswith('knn'):
                basic_cls = KNeighborsClassifier()
            else:
                raise Exception(f'unknown classifier {self.model_name}')
            one_vc_rest_model = OneVsRestClassifier(basic_cls)

            self.cv_scores = cross_validate(one_vc_rest_model, X, y,
                                            cv=5, return_estimator=True,
                                            return_train_score=True,
                                            scoring='f1_macro')

            saveit(self.cv_scores, self.ofile_cv_model_scores)

            self.model = self.cv_scores['estimator'][0]
            saveit(self.model, self.ofile_model_path)

    def predict(self, X_text: list):
        if self.model is None:
            raise Exception('Not fitted model')
        X = self.vect.transform(X_text)
        print('tfidf X shape', X.shape)
        return self.model.predict(X)
