import numpy as np
import lmdb
import sys
sys.path.append("../caffe/python")
from PIL import Image
import os
import caffe
from copy import deepcopy


HEIGHT = 500
WIDTH = 500

TRAIN_FILE_LIST = open("./data/500x500_images_mask/train.txt", "r").read().strip().split("\n")
TEST_FILE_LIST = open("/data/500x500_images_mask/test.txt", "r").read().strip().split("\n")
TRAIN_LABEL_FILE = open("./data/500x500_images_mask/mask_train.txt", "r").read().strip().split("\n")
TEST_LABEL_FILE = open("/data/500x500_images_mask/mask_test.txt", "r").read().strip().split("\n")

DIR = "/data/500x500_images_mask/"

def gen_input(lmdbname, file_list, phase='train'):
    X = np.zeros((len(file_list), 3, HEIGHT, WIDTH), dtype=np.float32)
    map_size = X.nbytes * 5

    env = lmdb.open(lmdbname, map_size=map_size)

    count = 0
    if phase == 'train':
        for i in file_list:
            if len(i) != 9:
                continue
            print count
            with env.begin(write=True) as txn:
                filename = os.path.join(DIR, "train_resize", i)
                m = np.asarray(Image.open(filename)).transpose((2, 0, 1))
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = m.shape[0]
                datum.height = m.shape[1]
                datum.width = m.shape[2]
                datum.data = m.tobytes()
                str_id = i
                txn.put(str_id.encode("ascii"), datum.SerializeToString())
                count += 1
    else:
        for i in file_list:
            if len(i) != 9:
                continue
            print count
            with env.begin(write=True) as txn:
                filename = os.path.join(DIR, "test_resize", i)
                m = np.asarray(Image.open(filename)).transpose((2, 0, 1))
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = m.shape[0]
                datum.height = m.shape[1]
                datum.width = m.shape[2]
                datum.data = m.tobytes()
                str_id = i
                txn.put(str_id.encode("ascii"), datum.SerializeToString())
                count += 1


def gen_output(lmdbname, file_list, phase='train'):
    X = np.zeros((len(file_list), 1, HEIGHT, WIDTH), dtype=np.uint8)
    map_size = X.nbytes * 3

    env = lmdb.open(lmdbname, map_size=map_size)
    
    count = 0
    if phase == "train":
        for i in file_list:
            print count
            with env.begin(write=True) as txn:
                filename = os.path.join(DIR, "mask_train", i)
                m = deepcopy(np.asarray(Image.open(filename)))
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = 1
                datum.height = m.shape[0]
                datum.width = m.shape[1]
                datum.data = m.tobytes()
                str_id = i
                txn.put(str_id.encode("ascii"), datum.SerializeToString())
                count += 1
    else:
        for i in file_list:
            print count
            with env.begin(write=True) as txn:
                filename = os.path.join(DIR, "mask_test", i)
                m = deepcopy(np.asarray(Image.open(filename)))
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = 1
                datum.height = m.shape[0]
                datum.width = m.shape[1]
                datum.data = m.tobytes()
                str_id = i
                txn.put(str_id.encode("ascii"), datum.SerializeToString())
                count += 1

gen_input("train_input_lmdb", TRAIN_FILE_LIST, phase='train')
gen_output("train_output_lmdb", TRAIN_LABEL_FILE, phase='train')

gen_input("test_input_lmdb", TEST_FILE_LIST, phase='test')
gen_output("test_output_lmdb", TEST_LABEL_FILE, phase='test')
