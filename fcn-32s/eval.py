import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/lancy/caffe/python")
from PIL import Image
from colorsys import hsv_to_rgb
import caffe
from copy import deepcopy


WIDTH, HEIGHT = 256, 256


caffe.set_device(1)
caffe.set_mode_gpu()
NUM = 200


filename = sys.argv[1]
im = Image.open(filename)
m = np.asarray(im, dtype=np.float32)
# m = m[:,:,::-1]
m -= np.array((104.00698793,116.66876762,122.67891434))
m = m.transpose((2, 0, 1))


labels = [np.array([i, j, k]) for i in range(0, 256, 80) for j in range(0, 256, 80) for k in range(0, 256, 80)]


net = caffe.Net(
    "deploy.prototxt", 
    "train_iter_10000.caffemodel",
    caffe.TEST)
net.blobs["data"].reshape(1, *m.shape)
net.blobs['data'].data[...] = m
net.forward()


'''
print im[0][0]
for i in range(WIDTH):
    for j in range(HEIGHT):
        if (im[0][0][i][j] > 0):
            print im[0][0][i][j]
            '''
# net.blobs["score"].data[0][0] = 0
m = deepcopy(net.blobs["score"].data[0]).transpose((1, 2, 0))
im = np.zeros((m.shape[0], m.shape[1], 3), np.uint8)


cnt = 0
stdv = m.max() * 4 / 10

for i in range(len(m)):
    for j in range(len(m[i])):
        if (m[i][j].max() > stdv):
            cnt += 1
            im[i][j] = labels[m[i][j].argmax()]

print cnt


img = Image.fromarray(im, "RGB")
img.save("show.png")


