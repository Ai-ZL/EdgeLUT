import os
import random
import sys

import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils import modcrop


class Provider(object):
    def __init__(self, batch_size, num_workers, scale, path, patch_size):
        self.data = DIV2K(scale, path, patch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return int(sys.maxsize)

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                         shuffle=False, drop_last=False, pin_memory=False))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]


class DIV2K(Dataset):
    def __init__(self, scale, path, patch_size, rigid_aug=True):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        fl = os.listdir(os.path.join(path, "DIV2K_data"))
        self.file_list = [f[:-4] for f in fl] 

        # need about 8GB shared memory "-v '--shm-size 8gb'" for docker container
        self.hr_cache = os.path.join(path, "cache_hr.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
            print("HR image cache to:", self.hr_cache)
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
        print("HR image cache from:", self.hr_cache)

        self.lr_cache = os.path.join(path, "cache_lr_{}.npy".format(self.scale))
        if not os.path.exists(self.lr_cache):
            self.cache_lr()
            print("LR image cache to:", self.lr_cache)
        self.lr_ims = np.load(self.lr_cache, allow_pickle=True).item()
        print("LR image cache from:", self.lr_cache)

    def cache_lr(self):
        lr_dict = dict()
        dataLR = os.path.join(self.path, "DIV2K_data")
        for f in self.file_list:
            lr_dict[f] = cv2.imread(os.path.join(dataLR, f+".png"))
            (h_all, w_all) = lr_dict[f].shape[:2]
            lr_dict[f] = cv2.resize(lr_dict[f],None,fx=1/3,fy=1/3,interpolation=cv2.INTER_NEAREST)
            (height, width) = lr_dict[f].shape[:2]
            (B,G,R) = cv2.split(lr_dict[f])
            bayer = np.empty((height, width), np.uint8)
            bayer[0::2, 0::2] = G[0::2, 0::2] # top left
            bayer[0::2, 1::2] = R[0::2, 1::2] # top right
            bayer[1::2, 0::2] = B[1::2, 0::2] # bottom left
            bayer[1::2, 1::2] = G[1::2, 1::2] # bottom right
            bayer = cv2.cvtColor(bayer, cv2.COLOR_GRAY2BGR)  # Convert from Grayscale to BGR (r=g=b for each pixel).
            bayer[0::2, 0::2, 0::2] = 0  # Green pixels - set the blue and the red planes to zero (and keep the green)
            bayer[0::2, 1::2, 0:2] = 0   # Red pixels - set the blue and the green planes to zero (and keep the red)
            bayer[1::2, 0::2, 1:] = 0    # Blue pixels - set the red and the green planes to zero (and keep the blue)
            bayer[1::2, 1::2, 0::2] = 0  # Green pixels - set the blue and the red planes to zero (and keep the green)
            bayer = cv2.resize(bayer, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            (h_b, w_b) = bayer.shape[:2]
            if h_b!=h_all and w_b!=w_all:
                lr_dict[f]=bayer[:h_all-h_b,:w_all-w_b,:]
            else:
                lr_dict[f]=bayer
            assert (lr_dict[f].shape[0] == h_all)
            assert (lr_dict[f].shape[1] == w_all)
        np.save(self.lr_cache, lr_dict, allow_pickle=True)

    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "DIV2K_data")
        for f in self.file_list:
            hr_dict[f] = cv2.imread(os.path.join(dataHR, f+".png"))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        lb = self.hr_ims[key]
        im = self.lr_ims[key]

        shape = im.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        # c = random.choice([0, 1, 2])

        lb = lb[i * self.scale:i * self.scale + self.sz * self.scale,
             j * self.scale:j * self.scale + self.sz * self.scale, :]
        im = im[i:i + self.sz, j:j + self.sz, :]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = np.transpose(lb.astype(np.float32) / 255.0, [2, 0, 1])
        im = np.transpose(im.astype(np.float32) / 255.0, [2, 0, 1])

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


class SRBenchmark(Dataset):
    def __init__(self, path, scale=4):
        super(SRBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        _ims_all = (5 + 29) * 2


        for dataset in ['classic5', 'LIVE1']:
            folder = os.path.join(path, dataset)
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = cv2.imread(os.path.join(path, dataset, files[i]))
                im_hr = modcrop(im_hr, scale)
                if len(im_hr.shape) == 2:
                    im_hr = np.expand_dims(im_hr, axis=2)

                    im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                key = dataset + '_' + files[i][:-4] # remove .png e.g. Set5_baby
                self.ims[key] = im_hr

                (h_all, w_all) = im_hr.shape[:2]
                im_lr = cv2.resize(im_hr,None,fx=1/3,fy=1/3,interpolation=cv2.INTER_NEAREST)
                (height, width) = im_lr.shape[:2]
                (B,G,R) = cv2.split(im_lr)
                bayer = np.empty((height, width), np.uint8)
                bayer[0::2, 0::2] = G[0::2, 0::2] # top left
                bayer[0::2, 1::2] = R[0::2, 1::2] # top right
                bayer[1::2, 0::2] = B[1::2, 0::2] # bottom left
                bayer[1::2, 1::2] = G[1::2, 1::2] # bottom right
                bayer = cv2.cvtColor(bayer, cv2.COLOR_GRAY2BGR)  # Convert from Grayscale to BGR (r=g=b for each pixel).
                bayer[0::2, 0::2, 0::2] = 0  # Green pixels - set the blue and the red planes to zero (and keep the green)
                bayer[0::2, 1::2, 0:2] = 0   # Red pixels - set the blue and the green planes to zero (and keep the red)
                bayer[1::2, 0::2, 1:] = 0    # Blue pixels - set the red and the green planes to zero (and keep the blue)
                bayer[1::2, 1::2, 0::2] = 0  # Green pixels - set the blue and the red planes to zero (and keep the green)
                bayer = cv2.resize(bayer, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                (h_b, w_b) = bayer.shape[:2]
                if h_b!=h_all and w_b!=w_all:
                    im_lr=bayer[:h_all-h_b,:w_all-w_b,:]
                else:
                    im_lr=bayer
                assert (im_lr.shape[0] == h_all)
                assert (im_lr.shape[1] == w_all)

                if len(im_lr.shape) == 2:
                    im_lr = np.expand_dims(im_lr, axis=2)

                    im_lr = np.concatenate([im_lr, im_lr, im_lr], axis=2)

                key = dataset + '_' + files[i][:-4] + 'n%d' % scale # e.g. Set5_babyn15
                self.ims[key] = im_lr

                assert (im_lr.shape[0] * scale == im_hr.shape[0])

                assert (im_lr.shape[1] * scale == im_hr.shape[1])
                assert (im_lr.shape[2] == im_hr.shape[2] == 3)

        assert (len(self.ims.keys()) == _ims_all)
