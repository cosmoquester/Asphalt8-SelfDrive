import tensorflow as tf
import numpy as np
from PIL import Image
from collections import deque
import pandas as pd
import cv2


input_dim = 5  # one-hot size
hidden_size = 16  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 5  # |ihello| == 6
learning_rate = 0.0001
last5_img = deque(maxlen=5)
last5_keyout = deque(maxlen=5)

def onehot(num):
    return np.identity(16)[num]

X = tf.placeholder(tf.float32, [5, 200, 400, 1])
Y = tf.placeholder(tf.float32, [5, 16])

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
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    prediction = tf.argmax(outputs, axis=1)

saver = tf.train.Saver()

with tf.Session() as sess:
    res = raw_input("Do you want addtional learning? (Name or Enter)")
    if res:
        saver.restore(sess, "./model/"+res+".ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    model_name = raw_input("Model_Name: ")
    
    try:
        for epoch in range(1):

            data_df = pd.read_csv('log.csv', names=['name','key_out'])
            names = data_df['name'].values
            name_i = 0
            # keyouts = [8, 1, 4, 2, 15, ...]
            key_out = [sum(np.array([8,4,2,1])*eval(x)) for x in data_df['key_out'].values]

            while len(last5_img) < 4 and len(names)>4:
                img = Image.open("./img/"+names[name_i])
                data = np.array( img, dtype='uint8' )
                last5_img.append(data)
                last5_keyout.append(onehot(key_out[name_i]))
                name_i+=1

            while(name_i < len(names)):
                img = Image.open("./img/"+names[name_i])
                data = np.array( img, dtype='uint8')
                last5_img.append(data)
                x_data = np.reshape(last5_img, [5, 200, 400, 1])
                last5_keyout.append(onehot(key_out[name_i]))

                print name_i, "/", len(names)
                sess.run(train, feed_dict={X:np.array(x_data, np.float32), Y:last5_keyout})
                name_i += 1
            print 'epoch', epoch+1
    finally:
        save_path = saver.save(sess, "./model/"+model_name+".ckpt")
