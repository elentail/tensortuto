import os,re,sys
import cv2
import numpy as np
import tensorflow as tf

filename = 'birds_train.tfrecord'

with tf.Session() as sess:
    file_queue = tf.train.string_input_producer([filename],name='queue') 

    reader = tf.TFRecordReader()

    _, tfrecord_serialized = reader.read(file_queue)
    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
    features={
    'image/class/label': tf.FixedLenFeature([], tf.int64),\
    'image/encoded': tf.FixedLenFeature([], tf.string),\
    }, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image/encoded'], tf.float32)
    print(image)
        


    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    #coord = tf.train.Coordinator() 
    #threads = tf.train.start_queue_runners(coord=coord)

    img = sess.run(image)

    #coord.request_stop()
    #coord.join(threads)

    
    rst = np.reshape(img,(290,290,3))
    rst = rst.astype(np.uint8)

    #cv2.imshow('RESULT',rst)
    #cv2.waitKey(0)
    print(rst)
