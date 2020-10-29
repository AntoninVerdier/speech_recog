import sys
import numpy as np 
import matplotlib.pyplot as plt
import IPython.display as ipd
import scipy.io.wavfile as wav

from PIL import Image
from network.branched_network_class import branched_network

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from tensorflow.python.framework import ops
ops.reset_default_graph()


# import the following to run demo_from_wav()
from pycochleagram import cochleagram as cgram 

## Some helper functions
def resample(example, new_size):
    im = Image.fromarray(example)
    resized_image = im.resize(new_size, resample=Image.ANTIALIAS)
    return np.array(resized_image)

def plot_cochleagram(cochleagram, title): 
    plt.figure(figsize=(6,3))
    plt.matshow(cochleagram.reshape(256,256), origin='lower',cmap=plt.cm.Blues, fignum=False, aspect='auto')
    plt.yticks([]); plt.xticks([]); plt.title(title); 
    
def play_wav(wav_f, sr, title):   
    print (title+':')
    ipd.display(ipd.Audio(wav_f, rate=sr))

def demo_pre_generated_cochleagram():
    ops.reset_default_graph()

    net_object = branched_network() # make network object
    word_key = np.load('demo_stim/logits_to_word_key.npy') #Load logits to word key 
    music_key = np.load('demo_stim/logits_to_genre_key.npy') #Load logits to genre key

    # example pre-generated speech cochleagram 
    example_cochleagram = np.load('demo_stim/example_cochleagram_0.npy') 
    plot_cochleagram(example_cochleagram,'Example speech cochleagram' )

    # run cochleagram through network and get logits for word branch
    logits = net_object.session.run(net_object.word_logits, feed_dict={net_object.x: example_cochleagram})

    # determine word branch prediction 
    prediction = word_key[np.argmax(logits, axis=1)]
    print ("Speech Example ... actual label: according  predicted_label: " + prediction[0] +'\n')
    
    # example pre-generated music cochleagram
    example_cochleagram_music = np.load('demo_stim/example_cochleagram_1.npy') 
    plot_cochleagram(example_cochleagram_music,'Example music cochleagram' )
    
    # run cochleagram through network and get logits for genre branch
    logits_music = net_object.session.run(net_object.genre_logits, 
                                          feed_dict={net_object.x: example_cochleagram_music})
    # note: throughout paper top-5 accuracy is reported for genre task
    prediction_music = (logits_music).argsort()[:,-5:][0][::-1] 
    print ("Music Example... actual label: "+ music_key[11]+ "  top-5 predicted_labels (in order of confidence): ")
    print ("\n"+ "; ".join(music_key[prediction_music]))

def generate_cochleagram(wav_f, sr, title):
    # define parameters
    n, sampling_rate = 50, 16000
    low_lim, hi_lim = 20, 8000
    sample_factor, pad_factor, downsample = 4, 2, 200
    nonlinearity, fft_mode, ret_mode = 'power', 'auto', 'envs'
    strict = True

    # create cochleagram
    c_gram = cgram.cochleagram(wav_f, sr, n, low_lim, hi_lim, 
                               sample_factor, pad_factor, downsample,
                               nonlinearity, fft_mode, ret_mode, strict)
    
    # rescale to [0,255]
    c_gram_rescaled =  255*(1-((np.max(c_gram)-c_gram)/np.ptp(c_gram)))
    
    # reshape to (256,256)
    c_gram_reshape_1 = np.reshape(c_gram_rescaled, (211,400))
    c_gram_reshape_2 = resample(c_gram_reshape_1,(256,256))
    
    plot_cochleagram(c_gram_reshape_2, title)

    # prepare to run through network -- i.e., flatten it
    c_gram_flatten = np.reshape(c_gram_reshape_2, (1, 256*256)) 
    
    return c_gram_flatten

def demo_from_wav():
    tf.reset_default_graph()
    net_object = branched_network()
    word_key = np.load('./demo_stim/logits_to_word_key.npy') # load logits to word key
    music_key = np.load('./demo_stim/logits_to_genre_key.npy') # load logits to word key 
    
    
    # generate cochleagram, then pass cochleagram through network and get logits for word branch
    
    ## Speech examples
    
    # example 1:
    sr, wav_f = wav.read('./demo_stim/example_1.wav') # note the sampling rate is 16000hz.
    play_wav(wav_f, sr, 'Example 1')
    c_gram = generate_cochleagram(wav_f, sr, 'Example 1')
    logits = net_object.session.run(net_object.word_logits, feed_dict={net_object.x: c_gram})
    prediction = word_key[np.argmax(logits, axis=1)]
    print ("Speech Example ... \n clean speech, actual label: Increasingly, predicted_label: " \
        + prediction[0] +'\n')
    
    ## Music examples 
    
    # example 6:
    sr, wav_f = wav.read('./demo_stim/example_6.wav') 
    play_wav(wav_f, sr, 'Example 6')
    c_gram = generate_cochleagram(wav_f, sr, 'Example 6')
    logits = net_object.session.run(net_object.genre_logits, feed_dict={net_object.x: c_gram})
    prediction = (logits).argsort()[:,-5:][0][::-1] 
    print ("Music Example ... \n Background: Auditory Scene, snr: -3db, actual label: " \
        + music_key[1] + ",\n top-5 predicted_labels (in order of confidence): \n " \
        + ";\n ".join(music_key[prediction]) + "\n")

demo_pre_generated_cochleagram()