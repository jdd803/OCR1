import cv2 as cv
import os
import numpy as np

img = cv.imread('./dataset/dataset/ch4_training_images/img_1.jpg')
gtbox = [[377,117,465,130,1],
         [493,115,519,131,1],
         [374,155,409,170,1],
         [492,151,551,170,1],
         [376,198,422,212,1],
         [494,189,539,206,1],
         [372,0,494,86,1]
]
mask =[[377,117,463,117,465,130,378,130,'Genaxis Theatre'],
       [493,115,519,115,519,131,493,131,'[06]'],
       [374,155,409,155,409,170,374,170,'###'],
       [492,151,551,151,551,170,492,170,'62-03'],
       [376,198,422,198,422,212,376,212,'Carpark'],
       [494,190,539,189,539,205,494,206,'###'],
       [374,1,494,0,492,85,372,86,'###'],
]
mask = np.array(mask)
gtbox = np.array(gtbox)
mask1 = np.reshape(mask[:,:8],(-1,4,2))
mask1 = mask1.astype(np.int32)
def draw(image,gtboxes):
    for i in range(len(gtboxes)):
        #cv.rectangle(image, (gtboxs[i][0],gtboxs[i][1]), (gtboxs[i][2],gtboxs[i][4]), (0, 255, 0), 1)
        line = gtboxes[i,:,:]
        pts = line.reshape([-1,1,2])

        cv.polylines(img, [pts], True, (0, 255, 255))
    cv.imshow("Image", image)
    cv.waitKey(0)
    pass
def draw1(image,gtboxes):
    for i in range(len(gtboxes)):
        print(gtboxes[i][0],gtboxes[i][1])
        cv.rectangle(image, (gtboxes[i][0],gtboxes[i][1]), (gtboxes[i][2],gtboxes[i][3]), (0, 255, 0), 1)
    cv.imshow("Image", image)
    cv.waitKey(0)
    pass

#draw(img,mask1)
draw1(img,gtbox)
