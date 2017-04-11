import glob as gl
import cPickle as cp
import numpy as np
import cv2

pics = np.empty([28,28,60000])
for i in range(60000):
    img = cv2.imread("/home/observer0724/Downloads/mnist/"+str(i+1)+".bmp")
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pics[:,:,i] = img_g
    print i
labelfile = open('/home/observer0724/Downloads/label.pkl')
label = cp.load(labelfile)
data = {}
data["label"] = label
data["pics"] = pics
output = open("/home/observer0724/Desktop/mnist.pkl","wb")
cp.dump(data,output)
output.close()