from PIL import ImageGrab
from getkeys import key_check
from sendkey import PressKey
from sendkey import ReleaseKey
from time import sleep
import tensorflow as tf
import cv2
from collections import deque
import numpy as np

hidden_size = 16  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence


A = 0x1E
S = 0x1F
D = 0x20
SPACE = 0x39

X = tf.placeholder(tf.float32, [5, 200, 400, 1])

with tf.variable_scope("cnn"):
    
    # conv : 5, 200, 400, 16
    conv = tf.layers.conv2d(inputs=X, filters=16, kernel_size=[4, 8], padding='SAME', activation=tf.nn.relu)
    # pool : 5, 50, 99, 16
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 4], padding='VALID', strides=1)
    # conv2 
    conv2 = tf.layers.conv2d(inputs=X, filters=64, kernel_size=[4, 8], padding='SAME', activation=tf.nn.relu)
    # pool2 
    pool2 = tf.layers.max_pooling2d(inputs=conv, pool_size=[4, 8], padding='VALID', strides=2)
    # conv3
    conv2 = tf.layers.conv2d(inputs=X, filters=128, kernel_size=[4, 8], padding='SAME', activation=tf.nn.relu)
    # pool3 
    pool3 = tf.layers.max_pooling2d(inputs=conv, pool_size=[4, 8], padding='VALID', strides=2)
    
    pool3 = tf.reshape(pool, [1, 5, -1])
    
with tf.variable_scope("rnn"):
    cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    #outputs : 1, 5, 50*99*16
    outputs, _states = tf.nn.dynamic_rnn(cell, pool3, initial_state=initial_state, dtype=tf.float32)
    
    X_for_softmax = tf.reshape(outputs, [-1, hidden_size])
    
    softmax_w = tf.get_variable("softmax_w", [hidden_size, hidden_size])
    softmax_b = tf.get_variable("softmax_b", [hidden_size])
    outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
    

    prediction = tf.argmax(outputs, axis=1)
    
saver = tf.train.Saver()
last5_img = deque(maxlen=5)


for i in range(5):
    print(i+1)
    sleep(1)


with tf.Session() as sess:
    saver.restore(sess, "./model/model_img4.ckpt")

    while True:
        snapshot = np.array(ImageGrab.grab())
        _img = cv2.resize(snapshot, (400,200),interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        data = np.array( img, dtype='uint8' )
        last5_img.append(data)

        if len(last5_img) < 5:
            continue
        
        x_data = np.reshape(np.array(last5_img), [5, 200, 400, 1])

        x = sess.run(prediction, feed_dict={X:np.array(x_data, np.float32)})[-1]
        acts = [x//8, (x%8)//4, (x%4)//2, x%2]

        ReleaseKey(A)
        ReleaseKey(D)
        ReleaseKey(SPACE)
        ReleaseKey(S)
        print(acts)
        if acts[1]:
            PressKey(S)
        if acts[0]:
            PressKey(A)
        if acts[2]:
            PressKey(D)
        if acts[3]:
            PressKey(SPACE)



        
