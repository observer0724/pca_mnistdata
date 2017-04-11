import cPickle as cp
from matplotlib import pyplot as plt

file = open("/home/observer0724/Desktop/mnist.pkl")
data = cp.load(file)
pics = data["pics"]
label = data["label"]
while(1):
    pic_num = int(raw_input("input pic number"))
    img = pics[:,:,pic_num]
    pic_label = str(label[pic_num])
    plt.imshow(img, cmap = "Greys_r")
    plt.suptitle(pic_label,fontsize = 12)
    plt.show()