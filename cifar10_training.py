#-*-coding:UTF-8-*-

from alexnet import alexnet_cnn
import time
import sys
import threading
import net_functions as tu
import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def train(epochs,
           batch_size,
           learning_rate,
           lmbda,
           momentum,
           resume,
           dropout,
           imagenet_path,
           display_step,
           test_step,
           summary_path,
           ckpt_path):
          
    NUM_IMAGES=5000
    NUM_BATCHES=int(float(NUM_IMAGES)/batch_size)
    x=tf.placeholder(tf.float32,[None,224,224,3])
    y=tf.placeholder(tf.float32,[None,100])
    lr=tf.placeholder(tf.float32)
    keep_prob=tf.placeholder(tf.float32)
    print('1')
          
    with tf.device('/cpu:0'):
      #构造一个存储队列，来缓存3倍batch的图片
        q=tf.FIFOQueue(batch_size*3,[tf.float32,tf.float32],[[224,224,3],[100]])
        enqueue_op=q.enqueue_many([x,y])
        x_b,y_b=q.dequeue_many(batch_size)
        
    pred=alexnet_cnn(x_b,dropout)
    
           
    with tf.name_scope('entropy_loss'):
        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y_b,name='cross_entropy'))
    with tf.name_scope('l2_loss'):
        l2_loss=tf.reduce_mean(lmbda*tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
        tf.summary.scalar('l2_loss',l2_loss)
    with tf.name_scope('loss'):
        loss=cross_entropy+l2_loss
        loss=tf.cast(loss,tf.float32)
        tf.summary.scalar('loss',loss)
    with tf.name_scope('accuracy'):
        correct=tf.equal(tf.argmax(pred,1),tf.argmax(y_b,1))
         #强制转化类型 tf.cast(x,dtype)
        accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
        tf.summary.scalar('accuracy',accuracy)


    global_step=tf.Variable(0,trainable=False)
    epoch=tf.div(global_step,batch_size)
    print('3')
    with tf.name_scope('optimizer'):
        optimizer=tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum).minimize(loss,global_step=global_step)
           
        merged=tf.summary.merge_all()
           
        saver=tf.train.Saver()
        coord=tf.train.Coordinator()
        init=tf.global_variables_initializer()
           
    with tf.Session(config=tf.ConfigProto()) as sess:
        if resume:
            saver.restore(sess,os.path.join(ckpt_path,'alexnet_cnn.ckpt'))
        else:
            sess.run(init)
        print('4')    
        def enqueue_batches():
            while not coord.should_stop():
                image_batches,label_batches=tu.read_batch(batch_size,imagenet_path)
                sess.run(enqueue_op,feed_dict={x:image_batches ,y:label_batches})
                #启动三个线程
        for i in range(3):
            t=threading.Thread(target=enqueue_batches)
            t.setDaemon(True)
            t.start()
        train_writer=tf.summary.FileWriter(os.path.join(summary_path,'train'),sess.graph)
                
        start_time=time.time()
        for e in range(sess.run(epoch),epochs):
            for i in range(NUM_BATCHES):
                _,step=sess.run([optimizer,global_step],feed_dict={lr:learning_rate,keep_prob:dropout})
                if step==170000 or step==350000:
                    learning_rate/=2

                if step%display_step==0:
                    acc,l=sess.run([accuracy,loss],feed_dict={lr:learning_rate,keep_prob:1.0})
                    print('Epoch:{:03d} step:{:09d}----Loss:{:.7f} Accuracy:{:.4f}'.format(e,step,l,acc))
                if step%test_step==0:
                    saver.save(sess,os.path.join(imagenet_path,'alexnet_cnn.ckpt'))
                        
        end_time=time.time()
        print('Elasped time:{}'.format(tu.format_time(end_time-start_time)))
        saver.save(sess,os.path.join(imagenet_path,'alexnet_cnn.ckpt'))
        coord.request_stop()


if __name__=='__main__':

    EPOCHS=90
    BATCH_SIZE=3
    LEARNING_RATE=1e-01
    MOMENTUM=0.9
    LAMBDA=5e-04
    DROP_OUT=0.5
    CKPT_PATH='/home/vatic/alexnet_tf/backup'
    IMAGENET_PATH='/home/vatic/alexnet_tf/data/cifar100_train'
    DISPLAY_STEP=10
    TEST_STEP=500
    SUMMARY='/home/vatic/alexnet_tf/summary'
    resume=False
    train(EPOCHS,
          BATCH_SIZE,
          LEARNING_RATE,
          LAMBDA,
          MOMENTUM,
          resume,
          DROP_OUT,
          IMAGENET_PATH,
          DISPLAY_STEP,
          TEST_STEP,
          SUMMARY,
          CKPT_PATH)
          
          
           
          

          
    
    
                  
                     
                        
                        
          
 


           
           
