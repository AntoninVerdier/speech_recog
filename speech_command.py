import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image


import keras
from keras import layers
from keras import models

from tqdm import tqdm
from scipy.io import wavfile
from multiprocessing import Process, Pool
import matplotlib.pyplot as plt
from pycochleagram import cochleagram as cgram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import forked_network as f

# Performance optimization
from timeit import default_timer as timer

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def load_data(label='bird'):
	data_path ='Data/train/audio/'
	labels = os.listdir(data_path)[:2]
	#labels.remove('_background_noise_') # Remove background noise for now
	all_samples = {}
	for label in labels:
		all_samples[label] = {}
		for file in os.listdir(os.path.join(data_path, label)):
			name = file[:-4] # Remove .wav extension
			samplerate, data = wavfile.read(os.path.join(data_path, label, file))
			if len(data) < 16000:
				data = np.append(data, [0]*(16000-len(data)))
			all_samples[label][name] = data

	return all_samples, labels

def resample(example, new_size):
	im = Image.fromarray(example)
	resized_image = im.resize(new_size, resample=Image.ANTIALIAS)
	return np.array(resized_image)

def plot_cochleagram(cochleagram, title):
	plt.figure(figsize=(6,3))
	plt.matshow(cochleagram.reshape(256,256), origin='lower',cmap=plt.cm.Blues, fignum=False, aspect='auto')
	plt.yticks([])
	plt.xticks([])
	plt.title(title)
	plt.show()


def generate_cochleagram(file, name):
	n, sampling_rate = 50, 16000
	low_lim, hi_lim = 20, 8000
	sample_factor, pad_factor, downsample = 4, 2, 400
	nonlinearity, fft_mode, ret_mode = 'power', 'auto', 'envs'
	strict = True
	# create cochleagram
	c_gram = cgram.cochleagram(file, sampling_rate, n, low_lim, hi_lim,
							   sample_factor, pad_factor, downsample,
							   nonlinearity, fft_mode, ret_mode, strict)

	# rescale to [0,255]
	c_gram_rescaled =  255*(1-((np.max(c_gram)-c_gram)/np.ptp(c_gram)))

	# reshape to (256,256)
	c_gram_reshape_1 = np.reshape(c_gram_rescaled, (211,400))
	c_gram_reshape_2 = resample(c_gram_reshape_1,(256,256))


	#plot_cochleagram(c_gram_reshape_2, title)

	# prepare to run through netwmap cork -- i.e., flatten it
	c_gram_flatten = np.reshape(c_gram_reshape_2, (1, 256*256))

	return c_gram_flatten

#all_samples, labels = load_data()
all_cochs = pickle.load(open('../Output/Cochleograms/all_coch.pkl', 'rb'))

# Unpack data
X, y = [], []
for label in all_cochs:
	for file in all_cochs[label]:
		X.append(all_cochs[label][file].reshape(256, 256, 1))
		y.append(label)

# Shape must be (n_images, x, y, n_channels) to enter conv2d
X, y = np.array(X), np.array(y)

# Convert categorical to numerical encoding
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# You can do validation split inside the model.fit function, use validation_split arg
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10298)

base = f.base_model()
speech = f.speech_branch()
genre = f.genre_branch()

optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

batch_size = 4
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

n_epochs = 10
switch = True
for epoch in range(n_epochs):
	print("\nStart of epoch %d" % (epoch,))
	for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
		if switch == True:
			switch==True
		if switch:
			current_model = keras.Model(inputs=base.input,outputs=speech(base(base.input)))
		else:
			current_model = keras.Model(inputs=base.input,outputs=genre(base(base.input)))
		
		with tf.GradientTape() as tape:
			logits = current_model(x_batch_train, training=True)  # Logits for this minibatch
			loss_value = loss_fn(y_batch_train, logits)

			grads = tape.gradient(loss_value, current_model.trainable_weights)
			optimizer.apply_gradients(zip(grads, current_model.trainable_weights))

		# Log every 200 batches.
		if not step % 10:
			print(
				"Training loss (for one batch) at step %d: %.4f"
				% (step, float(loss_value))
			)
			print("Seen so far: %s samples" % ((step + 1) * 64))
