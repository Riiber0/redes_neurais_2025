import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, GRU, MaxPooling1D, Dropout, TimeDistributed, Flatten, Dense, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt


class CNN_GRU:
    def __init__(self):
        self.model = None

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        #Bloco de duas Camadas CNN
        #Primeira camada convolucional
        c1 = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        c1 = Conv1D(64, 3, activation='relu', padding='same')(x)
        c1 = MaxPooling1D(max_pooling)(c1)
        c1 = Dropout(dropout)(c1)

        #Segunda camada convolucional
        c2 = Conv1D(128, 3, activation='relu', padding='same')(c1)
        c2 = MaxPooling1D(max_pooling)(c2)
        c2 = Dropout(dropout)(c2)

        #Bloco de duas camada GRU
        #Primeira camada GRU
        g3 = Bidirectional(GRU(128, return_sequences=True, dropout=dropout))(c2)
        g3 = BatchNormalization()(g3)

        #Segunda camada GRU
        g4 = Bidirectional(GRU(64, return_sequences=False, dropout=dropout))(g3)
        g4 = BatchNormalization()(g4)

        #Bloco de camada densa para regressao
        d5 = Dense(64, activation='relu')(g4)
        d5 = Dropout(dropout)(d5)

        #Camada de saida
        output_layer = Dense(2)(d5)

        model = Model(Inputs=input_layer, outputs=output_layer, name="CNN_GRU_regression")

        self.model = model

    def compile_model(self):
        self.model.compile(optmizer=keras.optmizer.Adam, loss='mse', 
                           metrics=[['mae', keras.metrics.RootMeanSquaredError(name='rmse')]])

    def model_summary(self):
        self.model.summary()

    def train_model(self, x_train, y_train, b_size):
        callbacks = [
                EarlyStopping(patience=10, restore_bst_weights=True),
                ReduceLROnPlateau(patience=10),
                ModelCheckpoint('best_model.h5',save_best_only=True)
        ]

        hist = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, 
                              batch_size=b_size, 
                              callbacks=callbacks, verbose=1)


