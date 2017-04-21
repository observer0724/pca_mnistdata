import pickle as cp
import numpy as np

def pca(matin,n):
    print ("PCA starts")
    meanmat = np.mean(matin,axis=0)
    diffmat = matin-meanmat

    norm_diffmat = np.empty((60000,1))
    for i in range(np.shape(norm_diffmat)[0]):
        vector = diffmat[i,:]
        lengthsq = 0
        for num in vector:
            lengthsq += num^2
        norm_diffmat[i,0] = lengthsq

    covmat = np.cov(diffmat, rowvar=0)
    eigval,eigvect = np.linalg.eig(covmat)
    eigindex = np.argsort(-eigval)
    outvects = eigvect[:,eigindex[0:n]]
    compressed_data = np.dot(diffmat,outvects)
    error = error(norm_diffmat,compressed_data,n)
    rebuilder = np.transpose(outvects)

    print ("PCA finished")
    print ("error is: %f"%error)
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

def error(norm, compressed,n):
    error_sheet = np.empty((60000,n))
    for i in range(np.shape(error_sheet)[0]):
        for j in range(np.shape(error_sheet)[1]):
            error_sheet[i,j] = (norm[i]-compressed[i,j]^2)^0.5
    error = sum(sum(error_sheet))/(60000*n)

def rebuild_error(rebuild,ori):
    diff = rebuild - ori
    distance_mat = np.empty((60000,1))
    for i in range(60000):
        for j in range(np.shape(diff)[1]):
            distance_mat[i,0] += diff[i,j]^2
        distance_mat[i,0] = distance_mat[i,0]^0.5

    rebuild_error = sum(distance_mat)/60000

file = open("mnist.pkl","rb")
data = cp.load(file,encoding="latin1")
pics = data["pics"]
label = data["label"]
m,n,l = np.shape(pics)
mat2d = squezze(pics)
compressed_mat,rebuilder,meanmat = pca(mat2d,3)
rebuild_pics = np.dot(compressed_mat,rebuilder)+meanmat
rebuild_error = rebuild_error(rebuild_pics,pics)
print ("rebuild error is: %f" % rebuild_error)
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
