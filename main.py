# -*- coding:UTF-8 -*-  
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

def make_dir(path):
    if not os.path.exists(path):
      os.makedirs(path)

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

make_dir('./result')
del_file('./result')
time_start=time.time()
detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

img = cv2.imread('./test_pic/test.jpg')

# run detector
results = detector.detect_face(img)

if results is not None:

    total_boxes = results[0]
    points = results[1]
    
    # extract aligned face chips
    chips = detector.extract_image_chips(img, points, 160, 0.37)
    
    make_dir('./result/mtcnn_output')

    for i, chip in enumerate(chips):
        # cv2.imshow('chip_'+str(i), chip)
        cv2.imwrite('./result/mtcnn_output/chip_'+str(i)+'.png', chip)

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,255 ,0),5)

    # for p in points:
    #     for i in range(5):
    #         cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

    cv2.imwrite("./result/mtcnn_output/mtcnn_result.png", draw)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    # cv2.imshow("detection result", draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# --------------
# test on camera
# --------------

# camera = cv2.VideoCapture(0)
# while True:
#     grab, frame = camera.read()
#     img = cv2.resize(frame, (320,180))
#
#     t1 = time.time()
#     results = detector.detect_face(img)
#     print 'time: ',time.time() - t1
#
#     if results is None:
#         continue
#
#     total_boxes = results[0]
#     points = results[1]
#
#     draw = img.copy()
#     for b in total_boxes:
#         cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
#
#     for p in points:
#         for i in range(5):
#             cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
#     cv2.imshow("detection result", draw)
#     cv2.waitKey(30)

