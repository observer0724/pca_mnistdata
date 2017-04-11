import cPickle as cp
import numpy as np
import cv2

def pca(matin,n):
    meanmat = np.mean(matin,axis=0)
    diffmat = matin-meanmat
    covmat = np.cov(diffmat, rowvar=0)
    eigval,eigvect = np.linalg.eig(covmat)
    eigindex = np.argsort(-eigval)
    outvects = eigvect[:,eigindex[0:n]]
    compressed_data = np.dot(diffmat,outvects)
    rebuilder = np.transpose(outvects)
    return compressed_data, rebuilder,meanmat

def squezze(pics):
    m,n,l  = np.shape(pics)
    mat2d = np.ndarray(shape = (m*n,l))
    for i in range(l):
        img = pics[:,:,i]
        vect = np.ndarray(shape = (m*n))
        for j in range(m):
            vect[j*n:(j+1)*n] = img[j,:]
            mat2d[:,i] = vect
    return np.transpose(mat2d)

def dragout(mat2d,m,n,l):
    pics = np.ndarray(shape = (m,n,l))
    for i in range(l):
        vect = mat2d[:,i]
        img = np.ndarray(shape=(m,n))
        for j in range(m):
            img[j,:] = vect[j*n:(j+1)*n]
        pics[:,:,i] = img
    return pics


file = open("/home/observer0724/Desktop/mnist.pkl")
data = cp.load(file)
pics = data["pics"]
label = data["label"]
m,n,l = np.shape(pics)
mat2d = squezze(pics)
compressed_mat,rebuilder,meanmat = pca(mat2d,3)
# rebuild_pics = np.dot(compressed_mat,rebuilder)+meanmat
# new_pics = dragout(np.transpose(rebuild_pics),m,n,l)
# for i in range(l):
#     img = new_pics[:,:,i]
#     cv2.imwrite("/home/observer0724/Desktop/mnist_compressed32/"+str(i)+".bmp",img)
file.close()

new_file = open("/home/observer0724/Desktop/compress_mnist_3.pkl","wb")
compressed_data = {}
compressed_data["data"] = compressed_mat
compressed_data["label"] = label

cp.dump(compressed_data,new_file)

new_file.close()