from __future__ import division
import cPickle as cp
import numpy as np
import time

def quality_func(matin, compressed,n):
    meanmat = np.mean(matin,axis=0)
    diffmat = matin-meanmat
    norm = np.empty((60000))
    for i in range(np.shape(norm)[0]):
        vector = diffmat[i,:]
        lengthsq = 0
        for num in vector:
            lengthsq += num**2
        norm[i] = lengthsq**0.5
    error_sheet = np.empty((60000,n))
    error_by_axle = np.empty((n))
    for i in range(np.shape(error_sheet)[0]):
        for j in range(np.shape(error_sheet)[1]):
            error_sheet[i,j] = (norm[i]**2-compressed[i,j]**2)**0.5
    for k in range(np.shape(error_by_axle)[0]):
        error_by_axle[k] = sum(error_sheet[:,k])/60000
    invert = []
    for l in range(np.shape(error_by_axle)[0]):
        invert.append(1/error_by_axle[l])
    quality = sum(invert)
    # print "errors are:"
    # print error_by_axle
    print "norm:"
    print norm
    return quality

def pca(matin,n):
    print ("PCA starts")
    st_time = time.time()
    meanmat = np.mean(matin,axis=0)

    diffmat = matin-meanmat
    print diffmat[0,:]
    print diffmat[200,:]
    print diffmat[20000,:]
    print diffmat[40000,:]

    covmat = np.cov(diffmat, rowvar=0)
    eigval,eigvect = np.linalg.eig(covmat)
    eigindex = np.argsort(-eigval)
    outvects = eigvect[:,eigindex[0:n]]
    compressed_data = np.dot(diffmat,outvects)
    rebuilder = np.transpose(outvects)
    ed_time = time.time()

    print ("PCA finished")
    # print ("duration time: %f"%(ed_time-st_time) )
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



def rebuild_error(rebuild,ori):
    diff = rebuild - ori
    distance_mat = np.empty((60000,1))
    for i in range(60000):
        for j in range(np.shape(diff)[1]):
            distance_mat[i,0] += diff[i,j]**2
        distance_mat[i,0] = distance_mat[i,0]**0.5

    rebuild_error = sum(distance_mat)/60000
    return rebuild_error

file = open("/home/yida/Desktop/pca_mnistdata/mnist.pkl","rb")
data = cp.load(file)
pics = data["pics"]
label = data["label"]
m,n,l = np.shape(pics)
mat2d = squezze(pics)
compressed_mat,rebuilder,meanmat = pca(mat2d,16)
rebuild_pics = np.dot(compressed_mat,rebuilder)+meanmat
# rebuild_error = rebuild_error(rebuild_pics,mat2d)
quality = quality_func(mat2d, compressed_mat,16)
print ("quality is: %f"% quality)
# print ("rebuild error is: %f" % rebuild_error)
# new_pics = dragout(np.transpose(rebuild_pics),m,n,l)
# for i in range(l):
#     img = new_pics[:,:,i]
#     cv2.imwrite("/home/observer0724/Desktop/mnist_compressed32/"+str(i)+".bmp",img)
file.close()

# new_file = open("/home/observer0724/Desktop/compress_mnist_3.pkl","wb")
# compressed_data = {}
# compressed_data["data"] = compressed_mat
# compressed_data["label"] = label
#
# cp.dump(compressed_data,new_file)
#
# new_file.close()
