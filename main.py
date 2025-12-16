import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import prepare_data, split_dataset, create_temporal_sequences
from model import CNN_GRU, baseModel
import os, sys
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODELS_PATH = './models/'

def create_model(model_type, name, x_train, y_train, x_val, y_val):
    if model_type == 'CNN':
        m = baseModel(name)
    elif model_type == 'CNN_GRU':
        m = CNN_GRU(name)

    m.create_regression_model(x_train.shape[1:])
    m.compile_model()
    m.model_summary()

    m.train_model(x_train, y_train, x_val, y_val, b_size=512)


def make_data(df, p_target, y_target):
    scaled_df = prepare_data(df, p_target, y_target)

    return scaled_df

def create_all(train_df, val_df):
    exp = []
    exp_1 = {'pred_time': 5,
            'p_target': 'pitch_pred_1',
            'y_target': 'yaw_pred_1',
            'target_id': 1
            }
    exp.append(exp_1)

    exp_2 = {'pred_time': 10,
            'p_target': 'pitch_pred_2',
            'y_target': 'yaw_pred_2',
            'target_id': 2
            }
    exp.append(exp_2)

    exp_3 = {'pred_time': 15,
            'p_target': 'pitch_pred_3',
            'y_target': 'yaw_pred_3',
            'target_id': 3
            }
    exp.append(exp_3)

    exp_4 = {'pred_time': 20,
            'p_target': 'pitch_pred_4',
            'y_target': 'yaw_pred_4',
            'target_id': 4
            }
    exp.append(exp_4)

    exp_5 = {'pred_time': 25,
            'p_target': 'pitch_pred_5',
            'y_target': 'yaw_pred_5',
            'target_id': 5
            }
    exp.append(exp_5)

    for i in exp:
        train_data = make_data(train_df, i['p_target'], i['y_target'])
        val_data = make_data(val_df, i['p_target'], i['y_target'])

        """
        x_train, y_train = create_temporal_sequences(train_data, i['pred_time'], True)
        x_val, y_val = create_temporal_sequences(val_data, i['pred_time'], True)
        create_model('CNN_GRU', f"CNN_GRU_{i['target_id']}", x_train, y_train, x_val, y_val)
        """

        x_train, y_train = create_temporal_sequences(train_data, i['pred_time'], False)
        x_val, y_val = create_temporal_sequences(val_data, i['pred_time'], False)
        create_model('CNN', f"CNN_{i['target_id']}", x_train, y_train, x_val, y_val)

def test_model(model_path, x_test, y_test):
    m = load_model(model_path)
    test_results = m.evaluate(x_test, y_test)

    print(m.name)
    print(f"Loss: {test_results[0]}")
    print(f"MAE: {test_results[1]}")
    print(f"RMSE: {test_results[2]}")

def test_all(test_df):

    models = glob.glob(MODELS_PATH+'*.h5')
    print(models)

    for model in models:
        if 'CNN_GRU' in model:
            target_id = model.split('_')[2]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = True
        else:
            target_id = model.split('_')[1]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = False

        test_data = make_data(test_df, p_target, y_target)

        x_test, y_test = create_temporal_sequences(test_data, pred_time, use_v)
        test_model(model, x_test, y_test)

if __name__ == '__main__':
    np.random.seed(9000)

    if len(sys.argv) < 2:
        print('help:\n\tcreate: create models and data\n\ttest: test models in models directory')
        sys.exit(0)

    if sys.argv[1] == 'create':
        raw_data = pd.read_csv('dataset_processed.csv')
        raw_data = raw_data.dropna()

        train_df, val_df, test_df = split_dataset(raw_data)
        test_df.to_csv('test_data.csv')

        create_all(train_df, val_df)

    elif sys.argv[1] == 'test':
        test_df = pd.read_csv('test_data.csv')
        test_all(test_df)





