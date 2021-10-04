import os
import time 
import random
import math
import argparse
import numpy as np
import utils
import tensorflow as tf
import model as nn_model
from tqdm import tqdm
from tensorflow import keras
from tensorflow.python.client import device_lib


random.seed(82)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=50, help="number epochs")
parser.add_argument("--batch_size", type=int, default=32, help="number batch_size default:64")

args = parser.parse_args()


print('epochs: {}\nbatch_size: {}'.format(args.epoch, args.batch_size))

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

tf.debugging.set_log_device_placement(False)

# set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

EPOCHS = args.epoch
BATCH_SIZE = args.batch_size


# set dir
DATA_DIR = './data'
BIN_DIR = os.path.join(DATA_DIR, 'bin')
OUTPUT_DIR = './output_model'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
            
input_list = utils.read_list(os.path.join(DATA_DIR,'training_list.txt'))
random.shuffle(input_list)
NUM_DATA = len(input_list)
print("Numbers of Training Data", NUM_DATA)

NUM_TRAIN = int(NUM_DATA*0.9) 
NUM_VALID = NUM_DATA-NUM_TRAIN

train_list= input_list[: NUM_TRAIN]
random.shuffle(train_list)
valid_list= input_list[NUM_TRAIN: ]

print('{} for training; {} for valid;'.format(NUM_TRAIN, NUM_VALID))        


InQSS = nn_model.InQSS()
# init model
model = InQSS.build()

print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={'avg_qua':'mse',
          'frame_qua':'mse',
          'avg_intell':'mse',
          'frame_intell':'mse'})


#loss logcosh
CALLBACKS = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR,'inqss.h5'),
        save_best_only=True,
        monitor='val_loss',
        verbose=1),
    keras.callbacks.TensorBoard(
        log_dir=os.path.join(OUTPUT_DIR,'tensorboard.log'),
        update_freq='epoch'), 
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0,
        patience=5,
        verbose=1)
]

# data generator
train_data = utils.data_generator(train_list, BIN_DIR, batch_size=BATCH_SIZE)
valid_data = utils.data_generator(valid_list, BIN_DIR, batch_size=BATCH_SIZE)


tr_steps = math.floor(NUM_TRAIN/BATCH_SIZE)
val_steps = math.floor(NUM_VALID/BATCH_SIZE)


# start fitting model
hist = model.fit_generator(train_data,
                           steps_per_epoch=tr_steps,
                           epochs=EPOCHS,
                           callbacks=CALLBACKS,
                           validation_data=valid_data,
                           validation_steps=val_steps,
                           verbose=1)
    



