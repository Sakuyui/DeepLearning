import numpy as np
import struct
import cv2
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time
from progressbar import *
import pandas as pd


##数据集存放位置的定义 Definition of the positiion of dataset
# 训练集文件
train_images_idx3_ubyte_file = 'dataset/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'dataset/train-labels.idx1-ubyte'
# 测试集文件
test_images_idx3_ubyte_file = 'dataset/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 'dataset/t10k-labels.idx1-ubyte'


#======================================================用于读取数据集的相关函数=======================================
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
   # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    #print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    #print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()

    progress = ProgressBar().start()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
           #print(progress.maxval)
            progress.update((i/60000)*100)
            #print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()

    progress.finish()
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    progress = ProgressBar().start()
    for i in range(num_images):
        #if (i + 1) % 10000 == 0:
            #print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    progress.finish()
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)
#======================================================用于读取数据集的相关函数=======================================


#######################################################Distance calculate Fucntions===================================
#计算某张img和目标所有图片的L2距离
def L2_Eucledian_Distance(img,d_imgs):
    r=d_imgs.shape[0]                #get rows' count (in there is 60000)
    converted_t_img=np.tile(img.reshape(1,784),(r,1))        #copy img's (shape=(1,784)) in row, to make it's shape is same to d_imgs (there d_imgs's shape is (60000,784))
    dist=np.sqrt(np.sum(np.power(converted_t_img-d_imgs,2),axis=1)) #calc L2 distance
    return dist

#计算某张img和目标所有图片的L1距离
def L1_Manhattan_Distance(img,d_imgs):
    r=d_imgs.shape[0]                #get rows' count (in there is 60000)
    converted_t_img=np.tile(img.reshape(1,784),(r,1))        #copy img's (shape=(1,784)) in row, to make it's shape is same to d_imgs (there d_imgs's shape is (60000,784))
    dist=np.sum(np.abs(converted_t_img-d_imgs),axis=1)    #calc L1 distance
    return dist

#######################################################Distance calculate Fucntions===================================




def K_Nearest(test_img, dataSet, labels, k=3,distance_func=L2_Eucledian_Distance):
    dataSetSize = dataSet.shape[0]#dataSet.shape[0]表示的是读取矩阵第一维度的长度，代表行数

    ds2=dataSet.reshape(dataSetSize,784)

    dist=L2_Eucledian_Distance(test_img,ds2)
    sortedDistIndicies = dist.argsort() #返回从小到大排序的索引

    classCount=np.zeros((10), np.int32)#10是代表10个类别
    for i in range(k): #统计前k个数据类的数量
        #print(sortedDistIndicies[i])
        voteIlabel = labels[sortedDistIndicies[i]]
        #print("投票标签"+str(voteIlabel))
        classCount[int(voteIlabel)] += 1
    max = 0
    id = 0


    for i in range(classCount.shape[0]):
        if classCount[i] >= max:
            max = classCount[i]
            id = i
    print(id)
    return id




if __name__ == '__main__':
    print("Load mnist dataset...")
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    print("Load finshed")


    count=0
    for i in range(test_images.shape[0]):

        predict=K_Nearest(test_img=test_images[i],k=5,dataSet=train_images,labels=train_labels)
        target=test_labels[i]
        #print("预测:"+str(predict)+" 实际:"+str(target))
        if(predict==target):
            count=count+1
        print(str(i+1)+"/"+str(test_images.shape[0])+"  正确数: "+str(count)+"  正确率"+str(count/np.double(i+1)))


    print('done')

