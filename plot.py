from matplotlib import pyplot as plt
import cPickle as cp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

file = open("/home/observer0724/Desktop/compress_mnist_3.pkl")
datas = cp.load(file)
label = datas["label"]
data = datas["data"]
x = []
y = []
z = []

for i in range(10):
    x_temp = []
    y_temp = []
    z_temp = []
    for j in range(60000):
        if i == int(label[j]):
            x_temp.append(data[j][0])
            y_temp.append(data[j][1])
            z_temp.append(data[j][2])

    x.append(x_temp)
    y.append(y_temp)
    z.append(z_temp)

figure = plt.figure()
ax = Axes3D(figure)
ax.scatter(x[0][:50],y[0][:50],z[0][:50],marker = "+",c = "r")
ax.scatter(x[1][:50],y[1][:50],z[1][:50],marker = "+",c = "g")
ax.scatter(x[2][:50],y[2][:50],z[2][:50],marker = "+",c = "b")
ax.scatter(x[3][:50],y[3][:50],z[3][:50],marker = "+",c = "k")
ax.scatter(x[4][:50],y[4][:50],z[4][:50],marker = "+",c = "y")
ax.scatter(x[5][:50],y[5][:50],z[5][:50],marker = "o",c = "r")
ax.scatter(x[6][:50],y[6][:50],z[6][:50],marker = "o",c = 'g')
ax.scatter(x[7][:50],y[7][:50],z[7][:50],marker = "o",c = "b")
ax.scatter(x[8][:50],y[8][:50],z[8][:50],marker = "o",c = "k")
ax.scatter(x[9][:50],y[9][:50],z[9][:50],marker = "o",c = "y")
plt.show()