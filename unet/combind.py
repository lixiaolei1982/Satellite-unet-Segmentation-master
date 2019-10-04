#coding=utf-8

import numpy as np
import cv2
from tqdm import tqdm

mask1_pool = ['testing1_plant_predict.png','testing1_building_predict.png',
              'testing1_water_predict.png','testing1_road_predict.png']

mask2_pool = ['testing2_plant_predict.png','testing2_building_predict.png',
              'testing2_water_predict.png','testing2_road_predict.png']

mask3_pool = ['testing3_plant_predict.png','testing3_building_predict.png',
              'testing3_water_predict.png','testing3_road_predict.png']              

## 0:none  1:vegetation   2:building   3:water   4:road

#after mask combind
img_sets = ['pre1.png','pre2.png','pre3.png']


def imgs_to_visualize(pred):
    assert (len(pred.shape) == 2)

    pred_images = np.empty((pred.shape[0], pred.shape[1], 3))

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i, j] == 1:
                pred_images[i, j, 0] = 159
                pred_images[i, j, 1] = 255
                pred_images[i, j, 2] = 84

            elif pred[i, j] == 2:
                pred_images[i, j, 0] = 34
                pred_images[i, j, 1] = 180
                pred_images[i, j, 2] = 238

            elif pred[i, j] == 3:
                pred_images[i, j, 0] = 255
                pred_images[i, j, 1] = 191
                pred_images[i, j, 2] = 0

            elif pred[i, j] == 4:
                pred_images[i, j, 0] = 38
                pred_images[i, j, 1] = 71
                pred_images[i, j, 2] = 139

    return pred_images

def combind_all_mask():
    for mask_num in tqdm(range(3)):
        if mask_num == 0:
            final_mask = np.zeros((5142,5664),np.uint8)#生成一个全黑全0图像,图片尺寸与原图相同
            final_mask_visualize = np.zeros((5142,5664,3),np.uint8)
        elif mask_num == 1:
            final_mask = np.zeros((2470,4011),np.uint8)
            final_mask_visualize = np.zeros((2470,4011, 3), np.uint8)
        elif mask_num == 2:
            final_mask = np.zeros((6116,3356),np.uint8)
            final_mask_visualize = np.zeros((6116,3356, 3), np.uint8)
        #final_mask = cv2.imread('final_1_8bits_predict.png',0)
        
        if mask_num == 0:
            mask_pool = mask1_pool
        elif mask_num == 1:
            mask_pool = mask2_pool
        elif mask_num == 2:
            mask_pool = mask3_pool
        final_name = img_sets[mask_num]
        for idx,name in enumerate(mask_pool):
            img = cv2.imread('./predict/'+name,0)
            height,width = img.shape
            label_value = idx+1  #coressponding labels value
            for i in tqdm(range(height)):    #priority:building>water>road>vegetation
                for j in range(width):
                    if img[i,j] == 255:
                        if label_value == 2:
                            final_mask[i,j] = label_value
                        elif label_value == 3 and final_mask[i,j] != 2:
                            final_mask[i,j] = label_value
                        elif label_value == 4 and final_mask[i,j] != 2 and final_mask[i,j] != 3:
                            final_mask[i,j] = label_value
                        elif label_value == 1 and final_mask[i,j] == 0:
                            final_mask[i,j] = label_value
        final_mask_visualize = imgs_to_visualize(final_mask)
        cv2.imwrite('./final_result/'+final_name,final_mask)
        cv2.imwrite('./final_result/'+'visualize_' + final_name, final_mask_visualize)




if __name__=='__main__':
    print('combinding mask...')
    combind_all_mask()
