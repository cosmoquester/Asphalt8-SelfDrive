from PIL import ImageGrab, Image
from getkeys import key_check
from sendkey import PressKey
from sendkey import ReleaseKey
from time import sleep
import tensorflow as tf
import cv2
from collections import deque
import numpy as np
import pandas as pd


hidden_size = 4  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
data_mean = 17.119790415671556
data_std = 63.36037302373175

A = 0x1E
S = 0x1F
D = 0x20
SPACE = 0x39

X = tf.placeholder(tf.float32, [5, 120, 300, 1])

with tf.variable_scope("cnn"):
    
    # conv : 5, 200, 400, 16
    conv = tf.layers.conv2d(inputs=X, filters=16, kernel_size=[4, 8], padding='SAME', activation=tf.nn.relu)
    # pool : 5, 50, 99, 16
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 4], padding='SAME', strides=1)
    # conv2 
    conv2 = tf.layers.conv2d(inputs=X, filters=64, kernel_size=[4, 8], padding='SAME', activation=tf.nn.relu)
    # pool2 
    pool2 = tf.layers.max_pooling2d(inputs=conv, pool_size=[4, 8], padding='SAME', strides=2)
    # conv3
    conv2 = tf.layers.conv2d(inputs=X, filters=128, kernel_size=[4, 8], padding='SAME', activation=tf.nn.relu)
    # pool3 
    pool3 = tf.layers.max_pooling2d(inputs=conv, pool_size=[4, 8], padding='SAME', strides=2)
    
    pool3 = tf.reshape(pool, [1, 5, -1])
    
with tf.variable_scope("rnn"):
    cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    
    initial_state = cell.zero_state(batch_size, tf.float32)
    #outputs : 1, 5, 50*99*16
    outputs, _states = tf.nn.dynamic_rnn(cell, pool3, initial_state=initial_state, dtype=tf.float32)
    
    X_for_sigmoid = tf.reshape(outputs, [-1, hidden_size])
    
    sigmoid_w = tf.get_variable("softmax_w", [hidden_size, hidden_size])
    sigmoid_b = tf.get_variable("softmax_b", [hidden_size])
    outputs = tf.sigmoid(tf.matmul(X_for_sigmoid, sigmoid_w) + sigmoid_b)
    
    prediction = outputs > 0.5
    
saver = tf.train.Saver()
last5_img = deque(maxlen=5)

model = input("Drive Model Name: ")
for i in range(5):
    print(i+1)
    sleep(1)


with tf.Session() as sess:
    saver.restore(sess, "./model/"+model+".ckpt")

    data_df = pd.read_csv('log.csv', names=['name','key_out'])
    names = data_df['name'].values
    name_i = 0
    key_out = [eval(x) for x in data_df['key_out'].values]

    while True:

        img = Image.open("./img/"+names[name_i])
##        snapshot = np.array(ImageGrab.grab())
##        _img = cv2.resize(snapshot, (400,200),interpolation=cv2.INTER_AREA)
##        img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
##        img = cv2.Canny(img, threshold1 = 250, threshold2=200)

        data = np.array( img, dtype='uint8' )
        data = (data-data_mean) / data_std
        last5_img.append(data)

        if len(last5_img) < 5:
            continue
        
        x_data = np.reshape(np.array(last5_img), [5, 120, 300, 1])

        x = sess.run(prediction, feed_dict={X:np.array(x_data, np.float32)})[-1]

##        ReleaseKey(A)
##        ReleaseKey(D)
##        ReleaseKey(SPACE)
##        ReleaseKey(S)
        print(x, key_out[name_i], [int(a*100)/100 for a in sess.run(outputs, feed_dict={X:np.array(x_data, np.float32)})[-1]])
        name_i += 1
##        if acts[1]:
##            PressKey(S)
##        if acts[0]:
##            PressKey(A)
##        if acts[2]:
##            PressKey(D)
##        if acts[3]:
##            PressKey(SPACE)



        
