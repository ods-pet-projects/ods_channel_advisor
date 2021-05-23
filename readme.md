# ods_channel_advisor
This project recommends channels for users by their welcome message.
Using previous user's activity history from dump ODS we trained NLP multi label classification model.

Project structure
- `scripts/prepare_posts.py` produces solid posts dataset with users id
- `scripts/prepare_users.py` produces `welcome` channel profiles
- `scripts/prepare_recommends.py` prepares users stats and create train dataset with `user_id`, `welcome_msg`, `top_channels`