import os

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')

from config import ROOT_DIR
from models.tfidf_model import TfidfModel
from scripts.prepare_posts import recommend_channels
from scripts.utils import loadit, saveit

target_channels = list(sorted(set(recommend_channels) - {'welcome'}))


def save_model(model, path):
    if not os.path.exists(path):
        model.save(path)
    else:
        print(f'path already exists {path}')


def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    print("accuracy", accuracy)
    for score_type in ['micro', 'macro', 'weighted', 'samples']:
        print(score_type)
        f1 = f1_score(y_val, predicted, average=score_type)
        precision = average_precision_score(y_val, predicted, average=score_type)
        print("\tF1-score", f1)
        print("\tprecision", precision)


def load_data(ifile_path, ofile_data):
    print(f'target channels {len(target_channels)}\n{target_channels}')

    if os.path.exists(ofile_data):
        print(f'found prepared data {ofile_data}')
        df = pd.read_csv(ofile_data)
        y = df.loc[:, target_channels]
    else:
        df = pd.read_csv(ifile_path).dropna(subset=['text'])
        y = (df.loc[:, target_channels] > 0).astype(int)
        df = df.copy()
        df = df.loc[:, ['text'] + target_channels]
        df.loc[:, target_channels] = (df.loc[:, target_channels] > 0).astype(int)
        df.to_csv(ofile_data)

    X_text = df['text']
    return X_text, y


def get_tfidf_logreg_model(model_name, X_text, y, model_cache_path):
    model_cache_path_obj = f'{model_cache_path}/model_obj'
    if os.path.exists(model_cache_path_obj):
        tfidf_model = loadit(model_cache_path_obj)
    else:
        tfidf_model = TfidfModel(model_name, model_cache_path)
        tfidf_model.fit(X_text, y)
        saveit(tfidf_model, model_cache_path_obj)
    return tfidf_model


def main():
    import sklearn
    print('sklearn version', sklearn.__version__)
    ifile_path = f'{ROOT_DIR}/data/users_activity.csv'
    ofile_data = f'{ROOT_DIR}/data/users_activity_data.csv'

    X_text, y = load_data(ifile_path, ofile_data)
    X_text_train, X_text_val, y_train, y_val = train_test_split(X_text, y, random_state=0)
    # print('X_text_val 0')
    # for i in range(30):
    #     print(X_text_val.iloc[i])
    #     print()
    #     print()
    #     print()
    #
    # return
    model_names = ['tfidf_logreg', 'tfidf_knn', 'count_logreg', 'count_knn', 'tfidf_ch_logreg, tfidf_ch_knn']
    for model_name in model_names:
        ofile_model_path = f'{ROOT_DIR}/cache/{model_name}'
        model = get_tfidf_logreg_model(model_name, X_text_train, y_train, ofile_model_path)
        print(f'{model_name} cv scores for {len(recommend_channels)} channels')
        print('cv_score f1 macro')
        print(model.cv_scores['train_score'])
        print(f'val score')
        y_pred = model.predict(X_text_val)
        print(f'{model_name} y_pred sum', y_pred.sum())
        print_evaluation_scores(y_val, y_pred)


def get_channels(y):
    ans_list = []
    for i, c in enumerate(target_channels):
        if y[i]:
            ans_list.append(f'{c}')
    return ans_list


def get_channels_text(y):
    ans_list = get_channels(y)
    return ', '.join(ans_list)


def get_tfidf_answer(text):
    # ofile_model_path = f'{ROOT_DIR}/cache/tfidf_logreg/model_obj'
    ofile_model_path = f'{ROOT_DIR}/cache/count_logreg/model_obj'
    model = loadit(ofile_model_path)
    y_pred = model.predict([text])
    channels = get_channels(y_pred[0])
    return channels


if __name__ == '__main__':
    main()
