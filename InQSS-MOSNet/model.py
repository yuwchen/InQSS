import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Softmax
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm

class InQSS(object):
    
    def __init__(self):

        print('Model init')
        
    def build(self):
        spec_input = keras.Input(shape=(None, 257))
        scat_input = keras.Input(shape=(None, 54+179))
        re_input = layers.Reshape((-1, 257, 1), input_shape=(-1, 257))(spec_input) 
        re_input_sc = layers.Reshape((-1, 54+179, 1), input_shape=(-1, 54+179))(scat_input) 
        

        # feature extraction of spectrogram
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
        conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
        re_shape = layers.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)


        # feature extraction of scattering coefficients
        conv1_sc = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input_sc)
        conv1_sc = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1_sc)
        conv1_sc = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1_sc)
        
        conv2_sc = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1_sc)
        conv2_sc = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2_sc)
        conv2_sc = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2_sc)
        
        conv3_sc = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2_sc)
        conv3_sc = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3_sc)
        conv3_sc = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3_sc)
        
        conv4_sc = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3_sc)
        conv4_sc = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4_sc)
        conv4_sc = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4_sc)
        
        re_shape_sc = layers.Reshape((-1, 3*128), input_shape=(-1, 3, 128))(conv4_sc)
        
        feature = tf.keras.layers.Concatenate()([re_shape, re_shape_sc])

        # quality prediction
        blstm_qua = Bidirectional(
            LSTM(256, return_sequences=True, dropout=0.3, 
                 recurrent_dropout=0.3, recurrent_constraint=max_norm(0.00001)), 
            merge_mode='concat')(feature)

        flatten_qua = TimeDistributed(layers.Flatten())(blstm_qua)
        dense_qua =TimeDistributed(Dense(128, activation='relu'))(flatten_qua)
        dense_qua = Dropout(0.3)(dense_qua)

        frame_score_qua = TimeDistributed(Dense(1), name='frame_qua')(dense_qua)
        average_score_qua = layers.GlobalAveragePooling1D(name='avg_qua')(frame_score_qua)
        

        # intelligibility prediction
        blstm_intell = Bidirectional(
            LSTM(256, return_sequences=True, dropout=0.3, 
                 recurrent_dropout=0.3, recurrent_constraint=max_norm(0.00001)), 
            merge_mode='concat')(feature)

        flatten_intell = TimeDistributed(layers.Flatten())(blstm_intell)
        dense_intell = TimeDistributed(Dense(128, activation='relu'))(flatten_intell)
        dense_intell = Dropout(0.3)(dense_intell)
    
        frame_score_intell = TimeDistributed(Dense(1), name='frame_intell')(dense_intell)
        average_score_intell = layers.GlobalAveragePooling1D(name='avg_intell')(frame_score_intell)


        model = Model(outputs=[average_score_qua, frame_score_qua, average_score_intell, frame_score_intell], inputs=[spec_input, scat_input])


        return model
    








