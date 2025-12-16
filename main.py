import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import split_dataset, create_temporal_sequences
from model import CNN_GRU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':

    raw_data = pd.read_csv('dataset_processed.csv')

    train_df, val_df, test_df = split_dataset(raw_data)
    raw_data = None

    x_train, y_train = create_temporal_sequences(train_df, ['pitch_pred_3', 'yaw_pred_3'])
    x_val, y_val = create_temporal_sequences(val_df, ['pitch_pred_3', 'yaw_pred_3'])
    x_test, y_test = create_temporal_sequences(test_df, ['pitch_pred_3', 'yaw_pred_3'])

    train_df = None
    val_df = None
    test_df = None

    m1 = CNN_GRU()

    m1.create_regression_model(x_train.shape[1:])
    m1.compile_model()
    m1.model_summary()

    hist = m1.train_model(x_train, y_train, x_val, x_val, b_size=1)
    test_results = m1.test_model(x_test, y_test)

    print(test_results)
