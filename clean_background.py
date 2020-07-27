# clean blackground
import cv2
import numpy as np
import imutils
import time
import glob

def clean_noise():
    count = 2370
    frame = cv2.imread('./clean/' +'output_' + str(count) + '.jpg')
    frame = imutils.resize(frame, width=800)
    image = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
    # 图像归一化
    fi = frame / 255.0
    # 伽马变换
    gamma = 0.4
    out = np.power(fi, gamma)
    # 以下只能灰階使用
    # 使用全局直方图均衡化
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equa = cv2.equalizeHist(grayimg)

    cv2.imshow('DeNoisingColor', image)
    # cv2.imshow('gamma', out)
    cv2.imshow('equa', equa)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def noise():
	count = 1001
	while True :        
		frame = cv2.imread('./clean/' +'output_' + str(count) + '.jpg')
		# frame = cv2.imread('./sample/' +'GetImage00' + str(count) + '.jpg')
		# frame = imutils.resize(frame, width=800) 
		image = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
		# kernel = np.array([(-0.125,-0.125,-0.125,-0.125,-0.125),
        #                   (-0.125, 0.25, 0.25, 0.25, -0.125),
        #                   (-0.125, 0.25, 1, 0.25, -0.125),
        #                   (-0.125, 0.25, 0.25, 0.25, -0.125),
        #                   (-0.125,-0.125,-0.125,-0.125,-0.125)]) 
        # result = cv2.filter2D(eq, ddepth=-1, kernel=kernel, 
        #                       anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
		
		cv2.imwrite('D:/tensorflow_object_detection/flask-video-stream-master/image/NLM/'+'output_' + str(count) + '.jpg', image)
		print("picture ", count," is finish!")
		count += 1
		
		# key = cv2.waitKey(1)
		if count == 92:
			break

	cv2.destroyAllWindows()

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    # print (len(channels))
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def light():
    for file in glob.glob("D:/tensorflow_object_detection/flask-video-stream-master/temp/*.jpg"):        
        # print('file ', file)
        frame = cv2.imread(file)
        # frame = cv2.imread('./sample/' +'GetImage00' + str(count) + '.jpg')
        # frame = imutils.resize(frame, width=800) 
        # image = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
        eq = hisEqulColor(frame)
        # kernel_size = 5
		# kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**2

        # kernel = np.array([(-0.125,-0.125,-0.125,-0.125,-0.125),
        #                   (-0.125, 0.25, 0.25, 0.25, -0.125),
        #                   (-0.125, 0.25, 1, 0.25, -0.125),
        #                   (-0.125, 0.25, 0.25, 0.25, -0.125),
        #                   (-0.125,-0.125,-0.125,-0.125,-0.125)]) 
        # result = cv2.filter2D(eq, ddepth=-1, kernel=kernel, 
        #                       anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
        # # 使用模糊後的圖抓取物件邊界
		# gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
		# blurred = cv2.GaussianBlur(gray, (11, 11), 0)
		# edged = cv2.Canny(blurred, 30, 150)
		# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		# coins = image.copy()
		# # 跟原始圖做對比畫黑色邊界
		# image_with_contours = cv2.drawContours(coins, cnts, -1, (255, 255, 255), 2)

        cv2.imwrite('D:/tensorflow_object_detection/augmentation/image/'+ file[-15:-4] + '_light.jpg', eq)
        print("picture ", file[-15:-4]," is finish!")
        
        


    cv2.destroyAllWindows()

def gray():
	for file in glob.glob("D:/tensorflow_object_detection/flask-video-stream-master/temp/*.jpg"):
		# print(file[-15:-4])
		frame = cv2.imread(file)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)		
		cv2.imwrite('D:/tensorflow_object_detection/augmentation/image/'+ file[-15:-4] + '_gray.jpg', gray)
		print("picture ", file[-15:-4]," is finish!")

def picture():
	for file in glob.glob("D:/tensorflow_object_detection/flask-video-stream-master/temp/*.jpg"):
		# print('file ', file)
		frame = cv2.imread(file)
		# frame = cv2.imread('./sample/' +'GetImage000' + str(count) + '.jpg') 
		# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
		frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		l_b = np.array([0, 0, 127])
		u_b = np.array([225, 255, 255])
		mask = cv2.inRange(hsv, l_b, u_b)
		res = cv2.bitwise_and(frame, frame, mask=mask)	
		cv2.imwrite('D:/tensorflow_object_detection/augmentation/image/'+ file[-15:-4] + '_black.jpg', res)
		print("picture ", file[-15:-4]," is finish!")
		

	cv2.destroyAllWindows()

if __name__=="__main__":
	# picture()
	# clean_noise()
	# noise()
	light()
	# gray()