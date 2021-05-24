import os
import glob
import json
import pandas as pd

DATA_DIR = '/Users/o/PycharmProjects/ods_help_bot/data2/opendatascience Slack export Mar 12 2015 - Sep 18 2020'
recommend_channels = [
        'deep_learning',
        'lang_python',
        'nlp',
        'edu_courses',
        'kaggle_crackers',
        'datasets',
        'lang_r',
        'visualization',
        'hardware',
        'big_data',
        'mltrainings_beginners',
        'cv',
        'devops',
        'trading',
        'reinforcement_learning',
        'blockchain',
        'satellite_imaging',
        'recommender_systems',
        'article_essence',
        'welcome',
    ]
posts_ofile_path = f'../data/ods_posts.csv'


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
    channel_stats_lines = []
    for channel_dir in sorted(glob.glob(f'{idir}/*')):
        if channel_dir.endswith('.json'):
            continue
        channel_files = glob.glob(f'{channel_dir}/*.json')
        channel = os.path.basename(channel_dir)
        channel_stats = dict(channel=channel, channel_files=len(channel_files))
        channel_stats_lines.append(channel_stats)
        if channel not in recommend_channels:
            continue

        print(channel_stats['channel'], channel_stats['channel_files'])
        for ifile_path in channel_files:
            with open(ifile_path, encoding='utf-8') as ifile:
                data_json_list = json.load(ifile)
                data_json_new_list = []
                for x in data_json_list:
                    x['channel'] = channel
                    data_json_new_list.append(x)
                lines.extend(data_json_new_list)

    df_channel_stats = pd.DataFrame(channel_stats_lines) \
        .sort_values('channel_files', ascending=False)
    print(df_channel_stats.head(50))

    df = pd.DataFrame(lines)
    # # %%
    # data = df.query('channel in @channels')
    print('init shape:', len(df))
    df = df.dropna(subset=['text'])
    df = df[~df['text'].str.contains(' has joined the channel')]
    df['text_len'] = df['text'].str.len()
    # df = df.query('text_len > 23')
    # need_cols = ['client_msg_id', 'channel', 'text']
    # df = df[need_cols]
    return df, df_channel_stats


@timeit
def main():
    idir = DATA_DIR
    ofile_stats_path = f'../data/ods_posts_stats.csv'
    # run over welcome slack and save all user_id
    # run over interesting channels and get user_id activity (post, comment)

    df, df_channel_stats = get_slack_df(idir)
    df.to_csv(posts_ofile_path, encoding='utf-8', index=False)
    df_channel_stats.to_csv(ofile_stats_path, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
