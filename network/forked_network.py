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
s = Conv2D(1024, [3, 3], [1, 1], padding='SAME', activation='relu')(x)
s = Conv2D(512, [3, 3], [1, 1], padding='SAME', activation='relu')(s)
s = MaxPooling2D([3, 3], [2, 2], padding='SAME')(s)
s = Flatten()(s)
s = Dropout(0.2)(s)
speech = Dense(n_words, activation='softmax')(s)

# Genre branch
g = Conv2D(1024, [3, 3], [1, 1], padding='SAME', activation='relu')(x)
g = Conv2D(512, [3, 3], [1, 1], padding='SAME', activation='relu')(g)
g = MaxPooling2D([3, 3], [2, 2], padding='SAME')(g)
g = Flatten()(g)
g = Dropout(0.2)(g)
genre = Dense(n_genres, activation='softmax')(g)

model = keras.Model(inputs=inputs, outputs=[speech, genre], name='kell_model')
print(model.summary())
keras.utils.plot_model(model, 'kell_prototype.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
			  optimizer=keras.optimizers.SGD(learning_rate=0.01),
			  metrics=['accuracy'])

""" Note to self on how they must have trained the network
1. They train the entire network with mixed input and let every sample alter both weights (unlikely)
2. They trained both branches before and after optimized the shared one after
3. They found a way to unactivate weight training for one type of input (unlikely given tensorflow 1 pipeline)
Go through methods
