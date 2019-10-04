#coding=utf-8

import cv2
import random
import numpy as np
from tqdm import tqdm

img_w = 256  
img_h = 256  

image_sets = ['1.png','2.png','3.png','4.png','5.png']

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪�?
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻�?
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb

def creat_dataset(image_num = 10000, mode = 'original',label_type='road'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        count_all = 0
        src_img = cv2.imread('C:/Users/admin/PycharmProjects/Satellite-Segmentation-master/data/src/' + image_sets[i])  # 3 channels
        label_img = cv2.imread('C:/Users/admin/PycharmProjects/Satellite-Segmentation-master/data/'+label_type+'/label/' + image_sets[i],cv2.IMREAD_GRAYSCALE)  # single channel
        X_height,X_width,_ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            key_filter_low = np.max(label_roi)* img_w * img_h * 0.0005  #0.005 100
            key_filter_high = np.max(label_roi)* img_w * img_h * 0.7
            count_all += 1
            if count_all > 10*image_each:  #防止死循环
                break
            if  np.sum(label_roi) < 10 or np.sum(label_roi) > key_filter_high or np.sum(label_roi) < key_filter_low:  #丢弃纯黑的训练样本
                continue

            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
            
            visualize = np.zeros((256,256)).astype(np.uint8)
            visualize = label_roi *50
            cv2.imwrite(('C:/Users/admin/PycharmProjects/Satellite-Segmentation-master/unet_train/'+label_type+'/visualize/%d.png' % g_count),visualize)
            cv2.imwrite(('C:/Users/admin/PycharmProjects/Satellite-Segmentation-master/unet_train/'+label_type+'/src/%d.png' % g_count),src_roi)
            cv2.imwrite(('C:/Users/admin/PycharmProjects/Satellite-Segmentation-master/unet_train/'+label_type+'/label/%d.png' % g_count),label_roi)
            count += 1 
            g_count += 1



if __name__=='__main__':

    #classes = ['plant', 'building', 'water', 'road']
    classes = ['plant', 'building']
    for j in range(len(classes)):
        creat_dataset(mode='augment',label_type = classes[j])
