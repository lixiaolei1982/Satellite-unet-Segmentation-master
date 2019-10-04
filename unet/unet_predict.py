import cv2
import numpy as np
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from tqdm import tqdm
from unet.lib.define_loss import balanced_cross_entropy,dice_coef




os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_SET = ['1.png','2.png','3.png']

image_size = 256
patch_height = 256
patch_width = 256
stride_height = 50
stride_width = 50

classes = [0,  1,  2,  3, 4]




def pred_to_imgs(pred, th = 0.5, mode="threshold"):
    assert (len(pred.shape)==2)

    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred_images[i,j]=pred[i,j]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if pred[i,j]>=th:
                    pred_images[i,j]=1
                else:
                    pred_images[i,j]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    return pred_images



def predict(label):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    #model = load_model(args["model"])
    #model = load_model('./unet_'+label +'_20.h5')
    model = load_model('./unet_' + label + '_20.h5',custom_objects={'balanced_cross_entropy_fixed':balanced_cross_entropy(),'dice_coef': dice_coef})
    #model = load_model('./unet_' + label + '_20.h5',custom_objects={'dice_coef': dice_coef})


    if(label=='water' or label=='road'):
        th = 0.5
    else:
        th = 0.5

    for n in tqdm(range(len(TEST_SET))):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread('C:/Users/admin/PycharmProjects/Satellite-Segmentation-master/test/' + path)
        image = np.array(image, dtype="float") / 255.0
        image_h,image_w,_ = image.shape
        image = img_to_array(image,data_format='channels_first')#（c,h,w）
        print(image.shape)
        leftover_h = (image_h-patch_height)%stride_height  #leftover on the h dim
        leftover_w = (image_w-patch_width)%stride_width  #leftover on the w dim
        if (leftover_h != 0):  #change dimension of img_h
            tmp_full_img = np.zeros((image.shape[0],image_h+(stride_height-leftover_h),image_w))
            tmp_full_img[0:image.shape[0],0:image_h,0:image_w] = image
            full_img = tmp_full_img
        if (leftover_w != 0):   #change dimension of img_w
            tmp_full_img = np.zeros((full_img.shape[0],full_img.shape[1],image_w+(stride_width - leftover_w)))
            tmp_full_img[0:full_img.shape[0],0:full_img.shape[1],0:image_w] = full_img
            full_img = tmp_full_img
        img_h = full_img.shape[1]  #height of the full image
        img_w = full_img.shape[2] #width of the full image
        N_patches_img = ((img_h-patch_height)//stride_height+1)*((img_w-patch_width)//stride_width+1)  #// --> division between integers
        print("Number of patches on h : " +str(((img_h-patch_height)//stride_height+1)))

        print("Number of patches on w : " +str(((img_w-patch_width)//stride_width+1)))

        print("number of patches per image: " +str(N_patches_img))
        print(full_img.shape)
        full_prob = np.zeros((img_h,img_w)).astype(np.float32)   #itialize to zero mega array with sum of Probabilities
        full_sum = np.zeros((img_h,img_w)).astype(np.float32)

        iter_tot = 0   #iter over the total number of patches (N_patches)
        for h in tqdm(range ((img_h-patch_height)//stride_height+1)): #loop over the full images

            for w in range((img_w-patch_width)//stride_width+1):

                patch = full_img[:,h*stride_height:(h*stride_height)+patch_height,w*stride_width:(w*stride_width)+patch_width]
                patch = np.expand_dims(patch, axis=0)
                pred = model.predict(patch,verbose=2)
                pred = pred.reshape((256,256))
                iter_tot +=1
                full_prob[h*stride_height:(h*stride_height)+patch_height,w*stride_width:(w*stride_width)+patch_width] +=pred[:,:]
                full_sum[h*stride_height:(h*stride_height)+patch_height,w*stride_width:(w*stride_width)+patch_width] += 1.0
        print(np.max(full_prob))
        assert(np.min(full_sum)>=1.0)
        print(iter_tot)
        assert (iter_tot == N_patches_img)
        final_avg = full_prob/full_sum
        pred_imgs = pred_to_imgs(final_avg,th).astype(np.uint8)
        pred_imgs = pred_imgs[0:image_h,0:image_w]
        pred_imgs = pred_imgs * 255
        cv2.imwrite('./predict/testing'+str(n+1) + '_' + label +'_predict.png',pred_imgs)





if __name__ == '__main__':

    labels = ['plant', 'building', 'water', 'road']
    #labels = ['road']
    #labels = ['plant']
    for i in range(len(labels)):
        predict(labels[i])