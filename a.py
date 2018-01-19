# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 19:46:24 2017
@author: HJL
"""

#%%

import tensorflow as tf
#import numpy as np
import os

#%%

# you need to change this to your data directory
train_dir = 'D:\\tensorflow\\train\\train\\'#Windows
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'#linux
#获取给定路径下图片名及其对应的标签
def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    images=[]
    labels=[]
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            images.append(file_dir + file)
            labels.append(0)
        else:
            images.append(file_dir + file)
            labels.append(1)
    return images, labels


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    #将python的list数据类型转换为tensorflow的数据类型
    #image = tf.cast(image, tf.string)
    #label = tf.cast(label, tf.int32)

    image = tf.convert_to_tensor(image, dtype=tf.string)
    label = tf.convert_to_tensor(label, dtype=tf.int32)
    
    # make an input queue  生成一个队列,shuffle=True即将图片打乱放入队列中
    input_queue = tf.train.slice_input_producer([image, label],shuffle=True)
    
    label = input_queue[1] #获取label对应的队列
    image_contents = tf.read_file(input_queue[0])#读取图片
    image = tf.image.decode_jpeg(image_contents, channels=3)#解码jpg格式图片
    
    ######################################
    # data argumentation should go to here
    ######################################
    #图片resize
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    #数据标准化
    image = tf.image.per_image_standardization(image)
    #Creates batches of tensors in tensors.
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 2, #线程数设置
                                                capacity = capacity) #队列中最多能容纳的元素
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)

    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


 
#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes

import matplotlib.pyplot as plt
BATCH_SIZE = 4
CAPACITY = 256
#图片resize后的大小
IMG_W = 208 
IMG_H = 208
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
train_dir = 'D:\\tensorflow\\train\\train\\'
image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
with tf.Session() as sess:#在会话中运行程序
    i = 0
    coord = tf.train.Coordinator()#线程协调者
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        #        Check if stop was requested.
        while not coord.should_stop() and i<1:
            
            img, label = sess.run([image_batch, label_batch])
            print(img[0,:,:,:])
            # just test one batch
            for j in range(BATCH_SIZE):
                print('label: %d' %label[j])
                #plt.imshow(img[j,:,:,:])
                #plt.show()
            i+=1
            
    except tf.errors.OutOfRangeError:#当读取完列队中所有数据时,抛出异常
        print('done!')
    finally:
        #Request that the threads stop.After this is called, calls to should_stop() will return True.
        coord.request_stop()
    coord.join(threads)


#%%


