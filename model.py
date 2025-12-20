import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, GRU, MaxPooling1D, Dropout, TimeDistributed, Flatten, Dense, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

MODELS_PATH='./models/'

class convModel():
    def __init__(self, name):
        self.model = None
        self.name = name

    @abstractmethod
    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        pass

    def compile_model(self):
        self.model.compile(optimizer=keras.optimizers.Adam(), loss='mse', 
                           metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')])

    def model_summary(self):
        self.model.summary()

    def train_model(self, x_train, y_train, x_val, y_val, b_size):
        callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(patience=10, verbose=1),
                ModelCheckpoint(MODELS_PATH+self.name+'_model.h5',save_best_only=True, verbose=1),
                CSVLogger(MODELS_PATH+self.name+'_training.log', separator=',', append=False)
        ]

        hist = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, 
                              batch_size=b_size, 
                              callbacks=callbacks, verbose=1)

        return hist

    def test_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def plot_model(self):
        keras.utils.plot_model(self.model, to_file=self.name
                               )

class CNN_GRU(convModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        #Bloco de duas Camadas CNN
        #Primeira camada convolucional
        c1 = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        c1 = Conv1D(64, 3, activation='relu', padding='same')(c1)
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
        output_layer = Dense(4)(d5)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self.model = model


class baseModel(convModel):
    def __init__(self, name):
        super().__init__(name)

    def create_regression_model(self, input_shape, dropout=0.15, max_pooling=2):
        input_layer = Input(shape=input_shape)

        #Primeira camada convolucional
        c1 = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        c1 = Conv1D(64, 3, activation='relu', padding='same')(c1)
        c1 = MaxPooling1D(max_pooling)(c1)
        c1 = Dropout(dropout)(c1)

        #Bloco de camada densa para regressao
        d2 = Dense(64, activation='relu')(c1)
        d2 = Dropout(dropout)(d2)

        #Camada de saida
        output_layer = Dense(4)(d2)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self.model = model

