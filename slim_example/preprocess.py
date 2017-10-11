
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import os,re,sys,cv2
import numpy as np
import tensorflow as tf


PREFIX_SIZE = (290,290)
VALIDATION_RATIO = 0.2
TRAIN_RATIO = 1 - VALIDATION_RATIO

tfExample = tf.train.Example


#
# http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
#
#


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



class PreParser(object):
    
    def __init__(self,*args,**kwargs):
    
        for k in kwargs:
            setattr(self,k,kwargs[k])


    def process(self):
        label , image_set = self._get_files_and_class()
        np.random.shuffle(image_set)

        train_len = int(len(image_set)*TRAIN_RATIO)   

        train_set = image_set[0:train_len]
        validation_set = image_set[train_len:]

        self._convert_data('train',label,train_set) 
        self._convert_data('validation',label,validation_set)

        return

    def _load_image_cv(self,image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img,PREFIX_SIZE,interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        return img

    def _convert_data(self,data_type,label_list,file_list):
        # open the TFRecords file
        with tf.python_io.TFRecordWriter(data_type) as record_writer:
            for f in file_list:

                class_key = f.split(os.path.sep)[-2]
                print(f)

                img = self._load_image_cv(f)

                label_index = label_list.index(class_key)
                # Create a feature
                feature = {\
                data_type+'/label': _int64_feature(label_index),\
                data_type+'/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))\
                }
                # Create an example protocol buffer
                example = tfExample(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                record_writer.write(example.SerializeToString())
        return

    def _get_files_and_class(self):

        _path = os.path.dirname(self.data_path)
        current_depth = _path.count(os.path.sep)

        class_list = []
        image_list = []

        for(root,dirs,files) in os.walk(_path):
            depth = root.count(os.path.sep) - current_depth
            # labels
            if(depth == 0):
                class_list += [d for d in dirs]
            # image files
            elif(depth == 1):
                image_list += [root+os.path.sep+f for f in files]
            else:
                break
        return class_list,image_list

if __name__ == '__main__':
    parser = PreParser(data_path='./')
    parser.process()
