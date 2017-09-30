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

    # train 에 따라 바뀌지 않고 학습시에 1씩 증가
    global_step = tf.Variable(0,trainable=False,name='STEP')


    # Layer1 scope
    with tf.name_scope('Layer1'):
        W1 = tf.Variable(tf.random_uniform([2,10],-1.,1.),name='W1')
        b1 = tf.Variable(tf.zeros([10]))
        L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),b1))

        tf.summary.histogram('Weight1',W1)

    with tf.name_scope('Layer2'):
        W2 = tf.Variable(tf.random_uniform([10,20],-1.,1.),name='W2')
        b2 = tf.Variable(tf.zeros([20]))
        L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),b2))

    with tf.name_scope('Output'):
        W3 = tf.Variable(tf.random_uniform([20,3],-1.,1.),name='W3')
        model = tf.matmul(L2,W3)

    with tf.name_scope('Optimizer'):
        #cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model),axis=1))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=model))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(cost,global_step=global_step)

        # 로그를 남기기 위해 scalar 형식의 cost 값 추적
        tf.summary.scalar('cost',cost)

    with tf.Session() as sess:

        # model 을 저장하기 위해 필수, tf.global_variables() Variable 로딩
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./model')

        if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()


        # 모든 로그 취합
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs',sess.graph)


        for step in range(100):
            error,_ = sess.run([cost,train_op],feed_dict={X:x_data,Y:y_data})
            gstep  = sess.run(global_step)
            print('STEP=[%d] , COST: [%.3f]'%(gstep,error))

            # 각 iteration 수행 마다 로그를 writer 에 저장
            summary = sess.run(merged ,feed_dict={X:x_data,Y:y_data})
            writer.add_summary(summary,global_step=gstep)

        saver.save(sess,'./model/dnn.ckpt',global_step=global_step)


        pred = tf.argmax(model,axis=1)
        target = tf.argmax(Y,1)

        print('예측값 : ',sess.run(pred,feed_dict={X:x_data,Y:y_data}))
        print('실제값 : ',sess.run(target,feed_dict={X:x_data,Y:y_data}))
        
        # equal 은 intiger 타입으로 float 타입으로 변환 필수
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred,target),tf.float32))
        print('정확도 : %.2f'%(sess.run(accuracy,feed_dict={X:x_data,Y:y_data})))

        


    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',help='input data')
    if(len(sys.argv) != 2):

        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()
    main(args.data_path)
