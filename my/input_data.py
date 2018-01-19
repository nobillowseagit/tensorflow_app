import tensorflow as tf
import numpy as np
import os
import math
 
#%%
 
# you need to change this to your data directory
###train_dir = '/home/ccf/Study/tensorflow/My-TensorFlow-tutorials-master/02_cats_vs_dogs/data/train/'
train_dir = 'D:\\tensorflow\\mydata\\'

 
def get_files(file_dir,ratio):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split('.')#这里file.split()函数内的参数py3和py2不同py3：name = file.split(seq='.')；py2：name = file.split('.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
 
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
 
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
 
    all_image_list =temp[:, 0]
    all_label_list = temp[:, 1]
 
    n_sample=int(len(all_label_list))  # 25000
    n_val=int(math.ceil(n_sample*ratio))  #
    n_train=int(n_sample-n_val)
 
    tra_images=all_image_list[0:n_train]
    tra_labels=all_label_list[0:n_train]
    tra_labels=[int(float(i)) for i in tra_labels]
    val_images=all_image_list[n_train:]
    val_labels=all_label_list[n_train:]
    val_labels=[int(float(i)) for i in val_labels]
    #label_list = [int(i) for i in label_list]
 
 
    return tra_images,tra_labels,val_images,val_labels
 
 
#%%
 


def get_files1(file_dir):  
    ''''' 
    Args: 
        file_dir: file directory 
    Returns: 
        list of images and labels 
    '''  
    cats = []  
    label_cats = []  
    dogs = []  
    label_dogs = []  
    for file in os.listdir(file_dir):  
        name = file.split(sep='.')  
        if name[0]=='cat':  
            cats.append(file_dir + file)  
            label_cats.append(0)  
        else:  
            dogs.append(file_dir + file)  
            label_dogs.append(1)  
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))  
      
    image_list = np.hstack((cats, dogs))  
    label_list = np.hstack((label_cats, label_dogs))  
      
    temp = np.array([image_list, label_list])  
    temp = temp.transpose()  
    np.random.shuffle(temp)  
      
    image_list = list(temp[:, 0])  
    label_list = list(temp[:, 1])  
    label_list = [int(i) for i in label_list]  
         
    return image_list, label_list  



def get_batch(image, label, image_W, image_H, batch_size, capacity):#capacity队列中最大容纳数据的个数
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
 
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
 
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
 
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
 
    ######################################
    # data argumentation should go to here
    ######################################
 
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
 
    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)
 
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
 
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
 
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
 
    return image_batch, label_batch
 
 
 
#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes
 
 
 
 
# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208
#
# train_dir = '/home/ccf/Study/tensorflow/My-TensorFlow-tutorials-master/01_cats_vs_dogs/data/train/'
#
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    try:
#        while not coord.should_stop() and i<1:
#
#            img, label = sess.run([image_batch, label_batch])
#
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
#