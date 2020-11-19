import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Lambda, MaxPooling2D, Flatten, Dense, Dropout


n_words = 2
n_genres = 41

inputs = keras.Input(shape=(256, 256, 1))

x = Conv2D(96, [9, 9], [3, 3], padding='SAME', activation='relu')(inputs)
# x = Lambda(tf.nn.lrn(depth_radius=2, bias=1, alpha=1e-3, beta=0.75))(x)
x = MaxPooling2D([3, 3], [2, 2], padding='SAME')(x)

x = Conv2D(256, [5, 5], [2, 2], padding='SAME', activation='relu')(x)
# x = Lambda(tf.nn.lrn(depth_radius=2, bias=1, alpha=1e-3, beta=0.75))(x)
x = MaxPooling2D([3, 3], [2, 2], padding='SAME')(x)

x = Conv2D(512, [3, 3], [1, 1], padding='SAME', activation='relu')(x)


# Need to create three different model => common, speech and branch

# Speech branch
x = Conv2D(1024, [3, 3], [1, 1], padding='SAME', activation='relu')(x)
x = Conv2D(512, [3, 3], [1, 1], padding='SAME', activation='relu')(x)
x = MaxPooling2D([3, 3], [2, 2], padding='SAME')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
speech = Dense(n_words, activation='softmax')(x)

# Genre branch
x = Conv2D(1024, [3, 3], [1, 1], padding='SAME', activation='relu')(x)
x = Conv2D(512, [3, 3], [1, 1], padding='SAME', activation='relu')(x)
x = MaxPooling2D([3, 3], [2, 2], padding='SAME')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
genre = Dense(n_genres, activation='softmax')(x)


model = keras.Model(inputs=inputs, outputs=[speech, genre], name='kell_model')
print(model.summary())
keras.utils.plot_model(model, 'kell_prototype.png', show_shapes=True)
