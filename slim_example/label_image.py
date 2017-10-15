"""
http://roadcom.tistory.com
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import os,re,sys
import argparse
import importlib
import cv2

import tensorflow as tf
from preprocessing import inception_preprocessing

slim = tf.contrib.slim

# prefix image size
image_size = 299

def run(args):
    data_path = args.data_path
    label_path = args.label_path

    model_path = args.model_path
    model_name = args.model_name
    model_scope = model_name +'_arg_scope'
    
    inception = importlib.import_module('nets.'+model_name)


    with tf.Graph().as_default():
        with slim.arg_scope(getattr(inception,model_scope)()):


            files = glob.glob(data_path+os.path.sep+"*.jpg")
            file_list = list()

            for idx,f in enumerate(files):
                f_string = tf.gfile.FastGFile(f, 'rb').read()
                test_img = tf.image.decode_jpeg(f_string, channels=3)

                processed_image = inception_preprocessing.preprocess_image(\
                        test_img, image_size, image_size, is_training=False)

                #processed_images = tf.expand_dims(processed_image, 0)
                file_list.append(os.path.basename(f))
                if(idx == 0):
                    processed_images = [processed_image]
                else:
                    processed_images.append(processed_image)

            processed_images = tf.stack(processed_images,axis=0)

            with open(label_path,'r') as rdata:
                names = dict()
                for row in rdata:
                    strip_row = row.strip()
                    split_row = strip_row.split(":")
                    if(len(split_row) == 2):
                        names[int(split_row[0])]=split_row[1]


            logits, _ = getattr(inception,model_name)(processed_images, num_classes=len(names), is_training=False)
            probabilities = tf.nn.softmax(logits)
            init_fn = slim.assign_from_checkpoint_fn(model_path, slim.get_model_variables('InceptionV3'))

            with tf.Session() as sess:
                init_fn(sess)

                np_image, probabilities = sess.run([processed_images, probabilities])

                print("\n========  DATA RESULT  =======\n")
                print("name\t"+"\t".join(names.values()))

                for idx,iter in enumerate(probabilities):
                    print(file_list[idx]+'\t' +'\t'.join([str(round(i,2)) for i in iter]))
                        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",help="the path to test images")
    parser.add_argument("--model_path")
    parser.add_argument("--model_name")
    parser.add_argument("--label_path")
    if(len(sys.argv) != 5):
        parser.print_help()
        parser.exit()
    args = parser.parse_args()
    run(args)

