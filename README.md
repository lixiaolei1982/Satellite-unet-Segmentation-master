# unet-Satellite-Segmentation

This is a satellite remote sensing segmentation project wirtten by Keras based on U-Net.

## main ideas
1. segmented by U-Net
2. loss-balanced_cross_entropy loss
3. image overlap ensemble



training original data
```
link：https://pan.baidu.com/s/1lQZDfriZpNL0yqWY_pvfKg 提取码：ubcs 
```


testing data
```
link：https://pan.baidu.com/s/1hMC3eQmJTnRotgnF_mTBwA 提取码：10me 
```



--Satellite-unet-Segmentation-master
   --data
         :training original data
   --test
         :testing data
   --unet
         :gen_dataset.py
         :unet_train.py
         :unet_predict.py
         :combind.py
         --lib
              :define_loss.py
         --predict
              :single class predict result
         --final_result
              :total class combind predict result
         
   --unet_train
         :training data


1step:
generater training data
```
python gen_dataset.py
```

2step:
unet training
```
python unet_train.py
```

3step:
single class predict (plant,building,water,road)
```
python unet_predict.py
```

4step:
total class combind predict result
```
python combind.py
```