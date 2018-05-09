import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2


input_dim = 5  # one-hot size
hidden_size = 4  # output from the LSTM. 5 to directly predict one-hot
batch_size = 100   # one sentence
sequence_length = 5  # |ihello| == 6
learning_rate = 0.0001
data_mean = 17.119790415671556
data_std = 63.36037302373175


X = tf.placeholder(tf.float32, [None, 200, 400, 1])
Y = tf.placeholder(tf.float32, [None, 4])

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
    
    flat = tf.reshape(pool3, [-1, 100*200*16])
    w = tf.get_variable("w", [hidden_size, hidden_size])
    b = tf.get_variable("b", [hidden_size])
    
    logit = tf.layers.dense(inputs=flat, units=4)
    outputs = tf.sigmoid(tf.matmul(logit, w) + b)
    
    loss = -tf.reduce_mean(Y*tf.log(outputs)+(1-Y)*(tf.log(1-outputs)))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


saver = tf.train.Saver()

with tf.Session() as sess:
    res = raw_input("Do you want addtional learning? (Name or Enter)")
    if res:
        saver.restore(sess, "./model/"+res+".ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    model_name = raw_input("Model_Name: ")
    
    try:
        for epoch in range(10):
            data_df = pd.read_csv('log.csv', names=['name','key_out'])
            names = data_df['name'].values
            # keyouts = [8, 1, 4, 2, 15, ...]
            key_out = [eval(x) for x in data_df['key_out'].values]
            
            while len(names):
                imgs = []
                offset = min(100, len(names))
                batch_x = names[0:offset]
                batch_y = key_out[0:offset]
                names = names[offset:]
                key_out = key_out[offset:]
                
                for name in batch_x:
                    img = Image.open("./img/"+name)
                    data = np.array( img, dtype='uint8')
                    data = data.reshape([200,400,1])
                    data = (data-data_mean) / data_std
                    imgs.append(data)

                sess.run(train, feed_dict={X:np.array(imgs, np.float32), Y:batch_y})
                print sess.run(loss, feed_dict={X:np.array(imgs, np.float32), Y:batch_y})
            print 'epoch', epoch+1, sess.run(loss, feed_dict={X:np.array(imgs, np.float32), Y:batch_y})
    finally:
        save_path = saver.save(sess, "./model/"+model_name+".ckpt")
