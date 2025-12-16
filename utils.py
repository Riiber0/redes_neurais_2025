import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_data(df, target_pitch, target_yaw):

    df = df.copy()

    df['pitch_sin'] = np.sin(df['pitch'])
    df['pitch_cos'] = np.cos(df['pitch'])
    df['yaw_sin'] = np.sin(df['yaw'])
    df['yaw_cos'] = np.cos(df['yaw'])

    velocity_scaler = StandardScaler()
    velocities = df[['pitch_v', 'yaw_v']].values
    velocities_scaled = velocity_scaler.fit_transform(velocities)

    df['pitch_v_scaled'] = velocities_scaled[:, 0]
    df['yaw_v_scaled'] = velocities_scaled[:, 1]

    df['pitch_pred_sin'] = np.sin(df[target_pitch])
    df['pitch_pred_cos'] = np.cos(df[target_pitch])
    df['yaw_pred_sin'] = np.sin(df[target_yaw])
    df['yaw_pred_cos'] = np.cos(df[target_yaw])

    return df


def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if train_ratio+val_ratio+test_ratio != 1:
        print('ratio not equal to 1')
        return

    id_cols = ['v_id', 'u_id', 'playback_time']

    features_cols = ['pitch', 'yaw', 'pitch_v', 'yaw_v']
    
    target_cols = ['pitch_pred_1', 'yaw_pred_1', 
                    'pitch_pred_2', 'yaw_pred_2',
                    'pitch_pred_3', 'yaw_pred_3',
                    'pitch_pred_4', 'yaw_pred_4',
                    'pitch_pred_5', 'yaw_pred_5']

    new_df = df[id_cols + features_cols + target_cols].copy()

    train_data, val_data, test_data = [], [], []

    for v_id in new_df['v_id'].unique():
        video_df = new_df[new_df['v_id'] == v_id].copy()

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

def create_temporal_sequences(df, pred_time, use_v=True):
    df = df.sort_values(['v_id', 'u_id', 'playback_time']).reset_index(drop=True)

    if use_v:
        features_cols = ['pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos', 
                         'pitch_v_scaled', 'yaw_v_scaled']
    else:
        features_cols = ['pitch_sin', 'pitch_cos', 'yaw_sin', 'yaw_cos']

    target_cols = ['pitch_pred_sin', 'pitch_pred_cos', 
                   'yaw_pred_sin', 'yaw_pred_cos']

    seq_len = 40 - pred_time
    
    x_sequences = []
    y_targets = []

    sessions = df.groupby(['v_id', 'u_id'])

    for (v_id, u_id), session_df in sessions:
        features = session_df[features_cols].values
        target = session_df[target_cols].values

        target_idx = seq_len
        for i in range(len(features) - seq_len):
            seq = features[i:i+seq_len-1]
            target_idx = i + seq_len - 1

            x_sequences.append(seq)
            y_targets.append(target[target_idx])

    x_seq = np.array(x_sequences, dtype=np.float64)
    y_seq = np.array(y_targets, dtype=np.float64)

    return x_seq, y_seq

