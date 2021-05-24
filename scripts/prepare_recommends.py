import pandas as pd
from tqdm import tqdm
from scripts.prepare_posts import posts_ofile_path, recommend_channels
from scripts.prepare_users import welcome_users_ofile_path

users_activity_ofile_path = '../data/users_activity.csv'

posts = pd.read_csv(posts_ofile_path)
users = pd.read_csv(welcome_users_ofile_path)
welcome_users = set(users['user'].unique())
users.drop_duplicates(subset=['user'], inplace=True)

users_posts = posts.query('user in @welcome_users')

users_activity = users_posts.groupby(['user', 'channel'], as_index=False)[['text']].count()


def get_user_pref(user, users_activity):
    line = dict(user=user)
    for channel in recommend_channels:
        rows = users_activity.query('user == @user and channel == @channel')
        if len(rows) == 0:
            line[channel] = 0
        else:
            line[channel] = rows['text'].values[0]
    return line


lines = []

for _, user_welcome_row in tqdm(users.iterrows()):
    user = user_welcome_row['user']
    user_welcome_text = user_welcome_row['text']
    user_welcome_text_len = user_welcome_row['text_len']

    user_pref_line = get_user_pref(user, users_activity)
    user_pref_line['text'] = user_welcome_text
    user_pref_line['text_len'] = user_welcome_text_len
    lines.append(user_pref_line)

users_activity_dataset = pd.DataFrame(lines)

print(users_activity.head())
print(users_activity_dataset)
print()
users_activity_dataset.to_csv(users_activity_ofile_path, index=False,  encoding='utf-8')
