import numpy as np
import pandas as pd

def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, target_cols=['pitch_pred_3', 'yaw_pred_3']):
    if train_ratio+val_ratio+test_ratio != 1:
        print('ratio not egual to 1')
        return

    new_df_cols = ['v_id', 'u_id', 'playback_time', 
                     'pitch', 'yaw', 'pitch_v', 'yaw_v',
                     'pitch_pred_3', 'yaw_pred_3']

    df = df[new_df_cols].copy()

    features_cols = ['pitch', 'yaw', 'pitch_v', 'yaw_v']

    train_data, val_data, test_data = [], [], []

    for v_id in df['v_id'].unique():
        video_df = df[df['v_id'] == v_id].copy()

        users = video_df['u_id'].unique()

        np.random.shuffle(users)
        n_users = len(users)

        n_train = max(1, int(n_users * train_ratio))
        n_val = max(1, int(n_users * val_ratio))
        n_test = n_users - n_train - n_val

        train_users = users[:n_train]
        val_users = users[n_train:n_train + n_val]
        test_users = users[n_train + n_val:]

        train_data.append(video_df[video_df['u_id'].isin(train_users)])
        val_data.append(video_df[video_df['u_id'].isin(val_users)])
        test_data.append(video_df[video_df['u_id'].isin(test_users)])

    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    return train_df, val_df, test_df

def create_temporal_sequences(df, target_cols):
    df = df.sort_values(['v_id', 'u_id', 'playback_time']).reset_index(drop=True)
    features_cols = ['pitch', 'yaw', 'pitch_v', 'yaw_v']

    pred_time = int(target_cols[0].split('_')[2]) * 5
    seq_len = 40 - pred_time
    
    x_sequences = []
    y_targets = []

    sessions = df.groupby(['v_id', 'u_id'])

    for (v_id, u_id), session_df in sessions:
        features = session_df[features_cols].values
        target = session_df[target_cols].values

        for i in range(len(features) - seq_len):
            seq = features[i:i+seq_len-1]

            x_sequences.append(seq)
            y_targets.append(target)

    x_seq = np.array(x_sequences, dtype=np.float64)
    y_seq = np.array(y_targets, dtype=np.float64)

    return x_seq, y_seq

