# pca_mnistdata

project of EECS405 "comparing Fastmap and PCA", This is the PCA part by Yida Zhou

readmnist.m convert mnist data into bmp pictures

label.py convert the label file into a pkl file the pca.py and plot.py can read.

use pic2pkl to convert the bmp pictures into the pickle data

pkl2pic.py to test if the pickle data is correct. Input a number, it will show the pic and its label

pca.py to do the pca reduction and produce the compressed data into pkl. Also it can compute the time cost and the stress

plot.py to plot a 3D space of data. It reads the 3D compressed pkl file generated from pca.py

plot2d.py to plot a 2D space. It reads the 2D compressed pkl file generated from pca.py
