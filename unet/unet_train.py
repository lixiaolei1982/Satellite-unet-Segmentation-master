#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import numpy as np  
from keras.models import Sequential  
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input
from keras.layers import concatenate,  core, Dropout
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder  
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
import matplotlib.pyplot as plt  
import cv2
import random
import os
from keras.optimizers import SGD
from unet.lib.define_loss import balanced_cross_entropy,dice_coef

from tqdm import tqdm  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7  
np.random.seed(seed)  
  

img_w = 256  
img_h = 256  
#有一个为背景  
#n_label = 4+1  
n_label = 1

image_sets = ['1.png','2.png','3.png']
 

def load_img(path, grayscale=False):
    if grayscale:
        #img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 1.0
        img = img/np.max(img) if np.max(img) != 0 else img  #归一化
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img



filepath ='C:/Users/admin/PycharmProjects/Satellite-Segmentation-master/unet_train/building/'

def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# data for training  
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'src/' + url)
            #img = img_to_array(img, data_format='channels_first')
            img = img_to_array(img,data_format='channels_first')
            train_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True) 
            label = img_to_array(label,data_format='channels_first')
            train_label.append(label)  
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label)  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateValidData...'
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img,data_format='channels_first')
            valid_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label,data_format='channels_first')
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label)  
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0  


#Define the neural network
def unet():

    inputs = Input((3, img_w, img_h))
    x = inputs
    depth = 4
    features = 32
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same',data_format='channels_first')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same',data_format='channels_first')(x)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same',data_format='channels_first')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same',data_format='channels_first')(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([skips[i], x], axis=1)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_first')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format='channels_first')(x)


    conv6 = Conv2D(n_label, (1, 1), padding='same',data_format='channels_first')(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)


    #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=1e-3), loss=[focal_loss(alpha=.25, gamma=2)], metrics=['accuracy'])

    model.compile(optimizer = Adam(lr=1e-4), loss=[balanced_cross_entropy()],
                  metrics=['accuracy',dice_coef])
    #model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy',dice_coef])
    return model

  

def train(label):

    EPOCHS = 20
    BS = 10

    model = unet()
    # =========== load pretrained_weights ========================
    pretrained_weights = './unet_'+label +'_20.h5'
    if (os.path.exists(pretrained_weights)):
        model.load_weights(pretrained_weights)

    #modelcheck = ModelCheckpoint('./unet_' + label + '_20.h5', monitor='val_acc', save_best_only=True, mode='max')
    modelcheck = ModelCheckpoint('./unet_'+ label +'_20.h5', monitor='val_acc', save_best_only=True, mode='max')

    callable = [modelcheck]  
    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)

    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=callable,max_q_size=1)  

    plt.style.use("ggplot")
    fig = plt.figure()
    N = EPOCHS
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    #ax1.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    ax1.set_ylabel('Loss')
    ax1.set_title("Training Loss and Accuracy on U-Net Satellite Seg")

    ax2 = ax1.twinx()  # this is the important function
    # ax2.plot(x, y2, 'r')
    ax2.plot(np.arange(0, N), H.history["dice_coef"], label="train_dice_coef")
    ax2.plot(np.arange(0, N), H.history["val_dice_coef"], label="val_dice_coef")

    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch #')
    plt.legend(loc="lower left")
    plt.savefig('./plot_' + label + '.png')




if __name__=='__main__':  

    #labels = ['plant', 'building', 'water', 'road']
    labels = ['plant', 'building']
    for i in range(len(labels)):

        filepath = 'C:/Users/admin/PycharmProjects/Satellite-Segmentation-master/unet_train/'+ labels[i] + '/'
        train(labels[i])
    #predict()  
