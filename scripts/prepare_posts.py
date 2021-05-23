import os
import glob
import json
import pandas as pd

DATA_DIR = '/Users/o/PycharmProjects/ods_help_bot/data2/opendatascience Slack export Mar 12 2015 - Sep 18 2020'


def timeit(func):
    def wrap(*args, **kwargs):
        start = pd.Timestamp.now()
        res = func(*args, **kwargs)
        elapsed = pd.Timestamp.now() - start
        print(f'elapsed {elapsed}')
        return res
    return wrap


def get_slack_df(idir):
    lines = []
    for channel_dir in sorted(glob.glob(f'{idir}/*')):
        if channel_dir.endswith('.json'):
            continue
        channel_files = glob.glob(f'{channel_dir}/*.json')
        channel = os.path.basename(channel_dir)
        print('{} {}'.format(channel, len(channel_files)))

        for ifile_path in channel_files:
            with open(ifile_path, encoding='utf-8') as ifile:
                data_json_list = json.load(ifile)
                data_json_new_list = []
                for x in data_json_list:
                    x['channel'] = channel
                    data_json_new_list.append(x)
                lines.extend(data_json_new_list)
    df = pd.DataFrame(lines)
    # channels = [
    #     'lang_python',
    #     'big_data',
    #     'datasets',
    #     'lang_r',
    #     'nlp',
    #     'deep_learning'
    # ]
    # # %%
    # data = df.query('channel in @channels')
    print('init shape:', len(df))
    df = df.dropna(subset=['text'])
    df = df[~df['text'].str.contains(' has joined the channel')]
    df['text_len'] = df['text'].str.len()
    # df = df.query('text_len > 23')
    # need_cols = ['client_msg_id', 'channel', 'text']
    # df = df[need_cols]
    return df


@timeit
def main():
    idir = DATA_DIR
    ofile_path = f'data/ods_posts.csv'
    # run over welcome slack and save all user_id
    # run over interesting channels and get user_id activity (post, comment)

    df = get_slack_df(idir)
    df.to_csv(ofile_path, encoding='utf-8')


if __name__ == '__main__':
    main()
