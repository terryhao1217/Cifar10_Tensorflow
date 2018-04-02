#-*-coding:UTF-8-*-
import os
import random
import tensorflow as tf
import numpy as np

from PIL import Image

###########Tensorflow standard operations wrappers#########

def weights(shape,name):
    w=tf.Variable(tf.truncated_normal(shape,stddev=5e-2),name=name)
    tf.add_to_collection('weights',w)
    return w

def biases(value,shape,name):
    b=tf.Variable(tf.constant(value,shape=shape),name=name)
    return b

def conv2d(x,W,stride,padding):
    return tf.nn.conv2d(x,W,strides=[1,stride[0],stride[1],1],padding=padding)

def max_pool2d(x,kernel,stride,padding):
    return tf.nn.max_pool(x,ksize=kernel,strides=stride,padding=padding)

def relu(x):
    return tf.nn.relu(x)

def batch_norm(x,shit,scale):
    #在batch_size维度下进行平均和方差计算
    epsilon=1e-3
    mean_val,var_val=tf.nn.moments(x,[0])
    return tf.nn.batch_normalization(x,mean_val,var_val,shift,scale.epsilon)

def lrn(x,depth_radius,bias,alpha,beta):
    return tf.nn.local_response_normalization(x,depth_radius,bias,alpha,beta)

    
def onehot(index):
    codes=np.zeros(100)
    codes[index]=1.0; #概率值
    return codes

def read_batch(batch_size,image_folder):
    batch_images=[]
    batch_labels=[]
    for i in range(batch_size):
        label_id=random.randint(1,100)
        batch_images.append(read_image(os.path.join(image_folder,str(label_id))))
        batch_labels.append(onehot(label_id-1)) #存储onehot形式的标签数据
    np.vstack(batch_images)
    np.vstack(batch_labels)
    
    return batch_images,batch_labels

def read_image(image_folder):
    image_path=os.path.join(image_folder,random.choice(os.listdir(image_folder)))
    return preprocess_image(image_path)

def preprocess_image(image_path):
    IMAGE_MEAN=[129.30,124.07,112.43]
    img=Image.open(image_path).convert('RGB')
    #resize成256×...,再进行裁剪
    if(img.size[0]<img.size[1]):
        h=int(float(256*img.size[1])/img.size[0])
        img=img.resize((256,h),Image.ANTIALIAS)
    else:
        w=int(float(256*img.size[0])/img.size[1])
        img=img.resize((w,256),Image.ANTIALIAS)  
  
    #crop to 224*224 size
    x=random.randint(0,img.size[0]-224)
    y=random.randint(0,img.size[1]-224)
    cropped_img=img.crop((x,y,x+224,y+224))
    cropped_im_array=np.array(cropped_img,dtype=np.float32)
    for i in range(3):
        cropped_im_array[:,:,i]-= IMAGE_MEAN[i]

    return cropped_im_array

def format_time(time):
    m,s=divmod(time,60)
    h,m=divmod(m,60)
    d,h=divmod(h,24)
    return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))

    
    

