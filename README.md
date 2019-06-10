# MTCNN_face_detection_and_alignment

## About

  This is a python/mxnet implementation of [Zhang](https://kpzhang93.github.io/)'s work **<Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks>**. it's fast and accurate,  see [link](https://github.com/kpzhang93/MTCNN_face_detection_alignment). 

  It should have **almost** the same output with the original work,  for mxnet fans and those can't afford matlab :)


## Requirement	  

- opencv 

  ​	I use cv2 for image io and resize(much faster than skimage), the input image's channel is acutally BGR

- mxnet 

  ​	**please update to the newest version, we need 'full' mode in Pooling operation**

Only tested on Linux and Mac

## Test

run:

 ``python main.py`` 

you can change `ctx` to `mx.gpu(0)` for faster detection

--- update 20161028 ---

by setting ``num_worker=4``  ``accurate_landmark=False`` we can reduce the detection time by 1/4-1/3, the bboxes are still the same, but we skip the last landmark fine-tune stage( mtcnn_v1 ). 

--- update 20161207 ---

add function `extract_face_chips`, examples:

![1](https://github.com/PLLLLLLL/mtcnn/blob/master/result/mtcnn_output/chip_0.png)
![2](https://github.com/PLLLLLLL/mtcnn/blob/master/result/mtcnn_output/chip_1.png)
![3](https://github.com/PLLLLLLL/mtcnn/blob/master/result/mtcnn_output/chip_2.png)
![4](https://github.com/PLLLLLLL/mtcnn/blob/master/result/mtcnn_output/chip_3.png)

see `mtcnn_detector.py` for the details about the parameters. this function use [dlib](http://dlib.net/)'s align strategy, which works well on profile images :) 
## Results

![big4](https://github.com/PLLLLLLL/mtcnn/blob/master/result/mtcnn_output/mtcnn_result.png)

## Reference

K. Zhang and Z. Zhang and Z. Li and Y. Qiao Joint,  Face Detection and Alignment Using Multitask Cascaded Convolutional Networks, IEEE Signal Processing Letters
# MTCNN
https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection
