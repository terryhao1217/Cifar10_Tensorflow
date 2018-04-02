#import cPickle as cp 
import numpy as np
from scipy.misc import imread
import os
import sys
from skimage import io
from matplotlib import image
def rgb_mean(image):
    r=np.mean(image[:,:,0])
    g=np.mean(image[:,:,1])
    b=np.mean(image[:,:,2])
    return [r,g,b]
    

def transform_cifar100Train(file_path):
    with open(file_path,'rb') as fo:
        cifar100_dict=cp.load(fo)
        prefix_name="image_"
        train_images=cifar100_dict['data']
        train_labels=cifar100_dict['fine_labels']
        [r_sum,g_sum,b_sum]=[0,0,0]
        NUM=50000
        for i in range(NUM):
            reshaped=np.reshape(train_images[i],[3,32,32])
            img=reshaped.transpose([1,2,0])
            [r,g,b]=rgb_mean(img)
            r_sum+=r;
            g_sum+=g;
            b_sum+=b;
            label_dir=train_labels[i]+1
            imsave('./data/cifar100_train/'+str(label_dir)+'/img_'+str(i)+'.jpg',img)
            print "...saving the  "+str(i)+'image' 
        print r_sum/NUM,g_sum/NUM,b_sum/NUM

def Caltech101_mean(file_path):
    [r,g,b]=[.0,.0,.0]
    NUM=0
    for i in range(1,100):
        
        for file_name in os.listdir(os.path.join(file_path,str(i))):
           
            img=image.imread(os.path.join(os.path.join(file_path,str(i)),file_name))
            
            r_mean=np.mean(img[:,:,0])
            g_mean=np.mean(img[:,:,1])
            b_mean=np.mean(img[:,:,2])
            r+=r_mean
            g+=g_mean
            b+=b_mean
            NUM=NUM+1
    print [r/NUM,g/NUM,b/NUM]

    

#cifar_100 [129.2162385574495, 123.98572316919191, 112.35082587594697]

    
Caltech101_mean('data/cifar100_train')
#transform_cifar100Train('./data/train')
