import pandas as pd
import numpy as np
import os

AGREGATED_DATASET_PATH='./AggregatedDataset/'

def get_angular_velocity(p1, p2):
    delta_p = p2 - p1
    speed = delta_p/0.1

    return speed

def get_ema(ema, v, alpha):
    return (1-alpha) * ema + alpha * v

datasets = os.listdir(AGREGATED_DATASET_PATH)
datasets.remove('readMe.txt')

for i in range(33, 60):
    datasets.remove(str(i) + '.txt')

print(datasets)

col_names = ['v_id', 'u_id', 'pitch', 'yaw', 'pitch_v', 'yaw_v', 
             'pitch_ema_1', 'yaw_ema_1', 'pitch_ema_2', 'yaw_ema_2', 'pitch_ema_3', 'yaw_ema_3',
             'pitch_pred_1', 'yaw_pred_1', 'pitch_pred_2', 'yaw_pred_2', 'pitch_pred_3', 
             'yaw_pred_3', 'pitch_pred_4', 'yaw_pred_4', 'pitch_pred_5', 'yaw_pred_5',
             'pitch_d_1', 'yaw_d_1', 'pitch_d_2', 'yaw_d_2', 'pitch_d_3', 
             'yaw_d_3', 'pitch_d_4', 'yaw_d_4', 'pitch_d_5', 'yaw_d_5']

final_df = pd.DataFrame(columns=col_names)

for file in datasets:
    print(file)
    dataset = pd.read_csv(AGREGATED_DATASET_PATH + file, sep=' ')

    user_id = 0
    pitch_ema = 0
    yaw_ema = 0
    video_id = file.split('.')[0]

    for r in range(0, len(dataset), 2):
        user_id += 1
        for c in range(0, dataset.shape[1]-1):
            pitch = np.float64(dataset.iloc[r, c])
            yaw = np.float64(dataset.iloc[r+1, c])

            future_pitch = np.float64(dataset.iloc[r, c+1])
            future_yaw = np.float64(dataset.iloc[r+1, c+1])

            pitch_v = get_angular_velocity(pitch, future_pitch)
            yaw_v = get_angular_velocity(yaw, future_yaw)

            pitch_ema_1 = get_ema(pitch_ema, pitch_v, np.float64(0.25))
            pitch_ema_2 = get_ema(pitch_ema, pitch_v, np.float64(0.5))
            pitch_ema_3 = get_ema(pitch_ema, pitch_v, np.float64(0.75))
            yaw_ema_1 = get_ema(yaw_ema, yaw_v, np.float64(0.25))
            yaw_ema_2 = get_ema(yaw_ema, yaw_v, np.float64(0.5))
            yaw_ema_3 = get_ema(yaw_ema, yaw_v, np.float64(0.75))

            #[0.5, 1, 1.5, 2, 2.5]
            pitch_pred = []
            yaw_pred = []

            pitch_d = []
            yaw_d = []

            for i in range(5, 26, 5):
                if c+i < len(dataset):
                    pitch_pred_t = np.float64(dataset.iloc[r, c+i])
                    yaw_pred_t = np.float64(dataset.iloc[r+1, c+i])

                    pitch_pred.append(pitch_pred_t)
                    yaw_pred.append(yaw_pred_t)

                    #displacement
                    pitch_d.append(pitch_pred_t - pitch)
                    yaw_d.append(yaw_pred_t - yaw)
                else:
                    pitch_pred.append(0)
                    yaw_pred.append(0)

                    pitch_d.append(0)
                    yaw_d.append(0)

            final_df.loc[len(final_df)] = [video_id, user_id, pitch, yaw, pitch_v, yaw_v, 
                                           pitch_ema_1, yaw_ema_1, pitch_ema_2, yaw_ema_2, pitch_ema_3, yaw_ema_3,
                                           pitch_pred[0], yaw_pred[0], pitch_pred[1], yaw_pred[1], pitch_pred[2], 
                                           yaw_pred[2], pitch_pred[3], yaw_pred[3], pitch_pred[4], yaw_pred[4],
                                           pitch_d[0], yaw_d[0], pitch_d[1], yaw_d[1], pitch_d[2], 
                                           yaw_d[2], pitch_d[3], yaw_d[3], pitch_d[4], yaw_d[4]]
            
    print(final_df)

    #os._exit(0)

final_df.to_csv('dataset_processed.csv', index=False)
