import pandas as pd

posts = pd.read_csv('../data/ods_posts.csv')
welcome_users_ofile_path = '../data/ods_users_welcome.csv'


def get_welcome_users(df):
    df_users = df.query('channel == "welcome" and reply_count and text != "This message was deleted."')
    return df_users


if __name__ == '__main__':
    df_users = get_welcome_users(posts)
    print('found welcome messages', len(df_users))
    print('found uniq user_id', len(df_users['user'].unique()))
    print(df_users.head())
    df_users.to_csv(welcome_users_ofile_path, index=False,  encoding='utf-8')
