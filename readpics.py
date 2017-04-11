import struct
import numpy as np








fname_lbl = "/home/observer0724/Downloads/train-labels-idx1-ubyte"
fname_img = "/home/observer0724/Downloads/train-images-idx3-ubyte"




with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)

# with open(fname_img, 'rb') as fimg:
#     magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
#     img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)


# print np.shape(img)
print lbl