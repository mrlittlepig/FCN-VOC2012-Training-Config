import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/lancy/caffe/python")
from PIL import Image
import caffe
from copy import deepcopy


def gen_net():
    caffe.set_device(1)
    caffe.set_mode_gpu()

    filename = '2007_000032.jpg'
    im = Image.open(filename)
    m = np.asarray(im, dtype=np.float32)
    m = m[:,:,::-1]
    m -= np.array((104.00698793,116.66876762,122.67891434))
    m = m.transpose((2, 0, 1))

    net = caffe.Net(
        "deploy.prototxt",
        #"train_iter_" + str(num) + ".caffemodel",
        #"/data/VGG16/caffemodel",
        "good.caffemodel",
        caffe.TRAIN)

    net.blobs["data"].reshape(1, *m.shape)
    net.blobs["data"].data[...] = m
    net.forward()
    return net

