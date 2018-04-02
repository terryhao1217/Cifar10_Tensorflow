#-*-coding:UTF-8-*-

import net_functions as tu
import tensorflow as tf
def cifar10_cnn(x,keep_prob,batch_size):
    
    with tf.name_scope('cifar_cnn') as scope:
        with tf.name_scope('cifar_cnn_conv1') as inner_scope:
            w1=tu.weights([5,5,3,64],name='w1')
            b1=tu.biases(0.0,[64],name='b1')
            conv1=tf.add(tu.conv2d(x,w1,stride=(1,1),padding='SAME'),b1)  #'SAME'不够则填充，‘VALID’多余则去掉
            conv1=tu.relu(conv1)
            norm1=tu.lrn(conv1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75)
            pool1=tu.max_pool2d(norm1,kernel=[1,3,3,1],stride=[1,2,2,1],padding='VALID')
 
        with tf.name_scope('cifar_cnn_conv2') as inner_scope:
            w2=tu.weights([5,5,64,64],name='w2')
            b2=tu.biases(0.1,[64],name='b2')
            conv2=tf.add(tu.conv2d(pool1,w2,stride=(1,1),padding='SAME'),b2)  #'SAME'不够则填充，‘VALID’多余则去掉
            conv2=tu.relu(conv2)
            norm2=tu.lrn(conv2,depth_radius=2,bias=1.0,alpha=2e-5,beta=0.75)
            pool2=tu.max_pool2d(norm2,kernel=[1,3,3,1],stride=[1,2,2,1],padding='VALID')
        
         
        
        with tf.name_scope('alexnet_cnn_conv3') as inner_scope:
            flattend=tf.reshape(pool2,[batch_size,-1])
            dim=flattend.get_shape()[1].value
            w6=tu.weights([dim,384],name='w6')
            b6=tu.biases(0.1,[384],name='b6')
            fc6=tf.add(tf.matmul(flattend,w6),b6)
            fc6=tu.relu(fc6)
            fc6=tf.nn.dropout(fc6,keep_prob)
        with tf.name_scope('alexnet_cnn_conv4') as inner_scope:
            w7=tu.weights([384,192],name='w7')
            b7=tu.biases(0.1,[192],name='b7')
            fc7=tf.add(tf.matmul(fc6,w7),b7)
            fc7=tu.relu(fc7)
            fc7=tf.nn.dropout(fc7,keep_prob)
        with tf.name_scope('alexnet_cnn_output') as inner_scope:
            w8=tu.weights([192,100],name='w8')
            b8=tu.biases(0.0,[100],name='b8')
            fc8=tf.add(tf.matmul(fc7,w8),b8)
           

    return fc8


 
         
