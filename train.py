#设置matplotlib将图片保存起来

import matplotlib
matplotlib.use('Agg')

#导包
import argparse
import os
import cv2
from imutils import paths
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from Lenet import LeNet


#定义参数解析器
def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-train", "--data_train", required=True, help="path to input data_train")
    ap.add_argument("-test", "--data_test", required=True, help="path to input data_test")
    ap.add_argument("-m", "--model", required=True, help='path to output the model')
    ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss")
    args = vars(ap.parse_args())
    return args

#参数初始化
EPOCHS = 35
INIT_LR = 1e-3
BATCH_SIZES = 32
NUM_CLASS = 62
NORM_SIZE = 32

#载入数据
def load_data(path):
    print("begin to load iamges...")
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(250)
    random.shuffle(imagePaths)
    for ip in imagePaths:
        image = cv2.imread(ip)
        image = cv2.resize(image, (NORM_SIZE, NORM_SIZE))
        image = img_to_array(image)
        data.append(image)

        label = int(ip.split(os.path.sep)[-2])
        labels.append(label)

    #归一化
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    #标签转化成哑编码形式
    labels = to_categorical(labels, num_classes=NUM_CLASS)
    return data, labels

#训练函数
def train(idg, X_train, X_test, y_train, y_test, args):
    print("compiling model ...")
    model = LeNet.build(width=NORM_SIZE, height=NORM_SIZE, depth=3, classes=NUM_CLASS)
    opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    #训练网络参数
    print("start to train network...")
    H = model.fit_generator(idg.flow(X_train, y_train, batch_size=BATCH_SIZES),
        validation_data=(X_test, y_test), steps_per_epoch=len(X_train)/BATCH_SIZES,
        epochs=EPOCHS, verbose=1)

    #保存网络模型
    print('save network...')
    model.save(args['model'])

    #绘制训练集的代价和准确率
    print("start to plot...")
    plt.style.use('ggplot')
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch#")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args['plot'])


if __name__=='__main__':
    args = args_parse()
    train_path = args['data_train']
    test_path = args['data_test']
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    #数据增强
    idg = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    train(idg, X_train, X_test, y_train, y_test, args)





