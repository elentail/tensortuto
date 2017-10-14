
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


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



class PreParser(object):
    
    def __init__(self,*args,**kwargs):
    
        for k in kwargs:
            setattr(self,k,kwargs[k])


    def process(self):
        label , image_set = self._get_files_and_class()

        with open('labels.txt','w') as wdata:
            for idx,name in enumerate(label):
                wdata.write('%d:%s\n'%(idx,name))
        np.random.shuffle(image_set)

        train_len = int(len(image_set)*TRAIN_RATIO)   

        train_set = image_set[0:train_len]
        validation_set = image_set[train_len:]


        self._write_tfrecord('train',label,train_set) 
        self._write_tfrecord('validation',label,validation_set)

        print("TRAIN_SIZE=%d,VALIDATION_SIZE=%d"%(len(train_set),len(validation_set)))
        return

    def _load_image_cv(self,image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img,PREFIX_SIZE,interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        return img

    def _read_tfrecord(self):
        """ To do implementation
        """
        return
    
    def _write_tfrecord(self,data_type,label_list,file_list):
    
        """ Convert a list of labels and files to TFRecord format.
        Args:
            data_type : train or validation type
            label_list : 
            file_list
        """
        record_name = '{}_{}.tfrecord'.format(self.dataset_name,data_type)
        # open the TFRecords file
        with tf.python_io.TFRecordWriter(record_name) as record_writer:
            for f in file_list:

                class_key = f.split(os.path.sep)[-2]
                img_format = (f.split(os.path.extsep)[-1])
                label_index = label_list.index(class_key)
                print(f)

                # to be modified
                #
                img = self._load_image_cv(f)
                ht,wd = np.shape(img)[:2]
                img_gfile = tf.gfile.FastGFile(f,'rb').read()

                """
                'image/encoded': bytes_feature(image_data),
                'image/format': bytes_feature(image_format),
                'image/class/label': int64_feature(class_id),
                'image/height': int64_feature(height),
                'image/width': int64_feature(width),
                """

                # Create a feature
                feature = {\
                'image/encoded': bytes_feature(img_gfile),\
                #'image/format': bytes_feature(img_format),\
                'image/format': bytes_feature(b'jpg'),\
                'image/class/label': int64_feature(label_index),\
                'image/height': int64_feature(ht),\
                'image/width': int64_feature(wd),\
                }
                # Create an example protocol buffer
                example = tfExample(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                record_writer.write(example.SerializeToString())
        return

    def _get_files_and_class(self):

        _path = os.path.dirname(self.dataset_dir)
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
    parser = PreParser(dataset_dir='./',dataset_name='birds')
    parser.process()
