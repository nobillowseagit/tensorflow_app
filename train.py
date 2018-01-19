import tensorflow as tf
import numpy as np
from PIL import Image
import random
IMAGE_MUMBER = 5000
EPOCH = 1
BATCH_SIZE = 100
IMAGE_PATH = "data/image/train/"
LABEL_PATH = "data/code_train_text.txt"
# 计算weight
def weigth_variable(shape):
    # stddev : 正态分布的标准差
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布
    return tf.Variable(initial)
# 计算biases
def bias_varibale(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 计算卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 定义池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHAR_SET_LEN = 10
xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  # 防止过拟合
x_image = tf.reshape(xs, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
# 训练网络
def code_cnn():
    # 第一个卷积层
    W_conv1 = weigth_variable([5, 5, 1, 32])
    b_conv1 = weigth_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # 14*14*32
    # 第二个卷积层
    W_conv2 = weigth_variable([5, 5, 32, 64])
    b_conv2 = weigth_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # 7*7*64
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)
    # 三层全连接层
    W_fc1 = weigth_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_varibale([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 防止过度拟合
    # 第四层全连接层
    W_fc2 = weigth_variable([1024, 10])
    b_fc2 = bias_varibale([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return prediction
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
# 文本转向量
def text2vec(text):
    text_len = len(text)
    vector = np.zeros(1 * CHAR_SET_LEN)
    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector
# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
# 生成一个训练batch
def get_next_batch(batch_size, each, images, labels):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, 10])
    def get_text_and_image(i, each):
        image_num = each * batch_size + i
        label = labels[image_num]
        image_path = images[image_num]
        captcha_image = Image.open(image_path)
        captcha_image = np.array(captcha_image)
        return label, captcha_image
    for i in range(batch_size):
        text, image = get_text_and_image(i, each)
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y
# 随机生成一个训练batch
def get_random_batch(batch_size, images, labels,IMAGE_MUMBER = IMAGE_MUMBER):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, 1 * CHAR_SET_LEN])
    def get_captcha_text_and_image(i):
        image_num = i
        label = labels[image_num]
        image_path = images[image_num]
        captcha_image = Image.open(image_path)
        captcha_image = np.array(captcha_image)
        return label, captcha_image
    for i in range(batch_size):
        text, image = get_captcha_text_and_image(random.randint(0, IMAGE_MUMBER - 1))
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y
# 计算准确率
def compute_accuracy(v_xs, v_ys, sess):  # 传入测试样本和对应的label
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result
prediction = code_cnn()
def train_code_cnn(image_paths, labels):
    # 定义网络
    global prediction
    # 计算loss cross_entropy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    saver = tf.train.Saver()      

    sess = tf.Session()
    # 初始化variable
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(EPOCH):
        # 每个epoch
        for each in range(int(IMAGE_MUMBER / BATCH_SIZE)):
            batch_x, batch_y = get_next_batch(BATCH_SIZE, each, image_paths, labels)
            _, loss_ = sess.run([train_step, cross_entropy]
                                , feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
            print("epoch: %d  iter: %d/%d   loss: %f"
                  % (epoch + 1, BATCH_SIZE * each, IMAGE_MUMBER, loss_))
        # 训练样本测试准确率
        test_iamge_path = "data/image/test/"
        test_labels_path = "data/code_test_text.txt"
        test_image_paths, test_labels = \
            get_image_path_labels(test_iamge_path, test_labels_path, 200)
        batch_x_test, batch_y_test = \
            get_random_batch(BATCH_SIZE, test_image_paths, test_labels,200)
        accuracy_test = compute_accuracy(batch_x_test, batch_y_test, sess)
        print("测试样本测试 epoch: %d  acc: %f" % (epoch + 1, accuracy_test))
        batch_x_test, batch_y_test = get_random_batch(BATCH_SIZE, image_paths, labels)
        accuracy = compute_accuracy(batch_x_test, batch_y_test, sess)
        print("训练样本测试 epoch: %d  acc: %f" % (epoch + 1, accuracy))
    saver.save(sess, "d:\\model1.ckpt") 
# 根据路径得到文本的内容
def getStrContent(path):
    return open(path, 'r', encoding="utf-8").read()
# 返回 训练样本路径的list 和 对应的标签用来以后训练
def get_image_path_labels(IMAGE_PATH=IMAGE_PATH, LABEL_PATH=LABEL_PATH, IMAGE_MUMBER=IMAGE_MUMBER):
    image_path = IMAGE_PATH
    label_path = LABEL_PATH
    image_paths = []
    for each in range(IMAGE_MUMBER):
        image_paths.append(image_path + str(each) + ".jpg")
    string = getStrContent(label_path)
    labels = string.split("#")
    return image_paths, labels
def main():
    # 得到训练样本路径list和标签的list
    image_paths, labels = get_image_path_labels()
    train_code_cnn(image_paths, labels)

 

if __name__ == '__main__':
    main()