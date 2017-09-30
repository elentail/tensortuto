import os,sys
import tensorflow as tf
import numpy as np
import argparse


def main(dpath):

    data = np.loadtxt(dpath,delimiter=',',dtype=np.float32)
    x_data = data[:,0:2]
    y_data = data[:,2:]

    print(x_data)
    print(y_data)


    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    global_step = tf.Variable(0,trainable=False,name='STEP')

    with tf.name_scope('Layer1'):
        W1 = tf.Variable(tf.random_uniform([2,10],-1.,1.))
        b1 = tf.Variable(tf.zeros([10]))
        L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),b1))

    with tf.name_scope('Layer2'):
        W2 = tf.Variable(tf.random_uniform([10,20],-1.,1.))
        b2 = tf.Variable(tf.zeros([20]))
        L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),b2))

    with tf.name_scope('Output'):
        W3 = tf.Variable(tf.random_uniform([20,3],-1.,1.))
        model = tf.matmul(L2,W3)

    with tf.name_scope('Optimizer'):
        #cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model),axis=1))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=model))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(cost,global_step=global_step)

        #tf.summary.scalar('cost',cost)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./model')

        if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()

        for step in range(100):
            error,_ = sess.run([cost,train_op],feed_dict={X:x_data,Y:y_data})
            gstep  = sess.run(global_step)
            print('STEP=[%d] , COST: [%.3f]'%(gstep,error))

        saver.save(sess,'./model/dnn.ckpt',global_step=global_step)


        pred = tf.argmax(model,axis=1)
        target = tf.argmax(Y,1)

        print('예측값 : ',sess.run(pred,feed_dict={X:x_data,Y:y_data}))
        print('실제값 : ',sess.run(target,feed_dict={X:x_data,Y:y_data}))

        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred,target),tf.float32))
        print('정홛도 : %.2f'%(sess.run(accuracy,feed_dict={X:x_data,Y:y_data})))

        


    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',help='input data')
    if(len(sys.argv) != 2):

        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()
    main(args.data_path)
