# -*- coding:utf-8 -*-    
  
from sys import path  
import numpy as np  
import tensorflow as tf  
import time  
import cv2  
from PIL import Image  
path.append('../..')  
#from common import extract_cifar10  
#from common import inspect_image  
  
  
#初始化单个卷积核上的参数  
def weight_variable(shape):  
    initial = tf.truncated_normal(shape, stddev=0.1)  
    return tf.Variable(initial)  
  
#初始化单个卷积核上的偏置值  
def bias_variable(shape):  
    initial = tf.constant(0.1, shape=shape)  
    return tf.Variable(initial)  
  
#卷积操作  
def conv2d(x, W):  
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  
  
  
  
def main():  
    #定义会话  

    sess = tf.InteractiveSession()  
      
    #声明输入图片数据，类别  
    x = tf.placeholder('float',[None,32,32,3])  
    y_ = tf.placeholder('float',[None,10])  
  
    #第一层卷积层  
    W_conv1 = weight_variable([5, 5, 3, 64])  
    b_conv1 = bias_variable([64])  
    #进行卷积操作，并添加relu激活函数  
    conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)  
    # pool1  
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')  
    # norm1  
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')  
  
  
    #第二层卷积层  
    W_conv2 = weight_variable([5,5,64,64])  
    b_conv2 = bias_variable([64])  
    conv2 = tf.nn.relu(conv2d(norm1,W_conv2) + b_conv2)  
    # norm2  
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')  
    # pool2  
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')  
  
    #全连接层  
    #权值参数  
    W_fc1 = weight_variable([8*8*64,384])  
    #偏置值  
    b_fc1 = bias_variable([384])  
    #将卷积的产出展开  
    pool2_flat = tf.reshape(pool2,[-1,8*8*64])  
    #神经网络计算，并添加relu激活函数  
    fc1 = tf.nn.relu(tf.matmul(pool2_flat,W_fc1) + b_fc1)  
      
    #全连接第二层  
    #权值参数  
    W_fc2 = weight_variable([384,192])  
    #偏置值  
    b_fc2 = bias_variable([192])  
    #神经网络计算，并添加relu激活函数  
    fc2 = tf.nn.relu(tf.matmul(fc1,W_fc2) + b_fc2)  
  
  
    #输出层，使用softmax进行多分类  
    W_fc2 = weight_variable([192,10])  
    b_fc2 = bias_variable([10])  
    y_conv=tf.maximum(tf.nn.softmax(tf.matmul(fc2, W_fc2) + b_fc2),1e-30)  
  
    #  


    #input  
    im = Image.open('dog8.jpg')  
    im.show()  
    im = im.resize((32,32))  
    # r , g , b = im.split()  
    # im = Image.merge("RGB" , (r,g,b))  
    print(im.size , im.mode)  
  
    im = np.array(im).astype(np.float32)  
    im = np.reshape(im , [-1,32*32*3])  
    im = (im - (255 / 2.0)) / 255  
    batch_xs = np.reshape(im , [-1,32,32,3])  
    #print batch_xs  
    #获取cifar10数据  
    # cifar10_data_set = extract_cifar10.Cifar10DataSet('../../data/')  
    # batch_xs, batch_ys = cifar10_data_set.next_train_batch(1)  
    # print batch_ys  



    saver = tf.train.Saver()

    output = sess.run(y_conv , feed_dict={x:batch_xs})  

    saver.restore(sess , 'd:\\checkpoint')  


    print(output)  
    print('the out put is :' , np.argmax(output))
    #关闭会话  
    sess.close()  
  
if __name__ == '__main__':  
    main()  