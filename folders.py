import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io as scio
import numpy as np
import csv
from openpyxl import load_workbook


class CuzFolder(data.Dataset):

    def __init__(self, root, index, transform):
        name_mat = scio.loadmat("E:\CUZ2021\\imgs.mat")
        label_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\OBSCsn.mat")
        imgname = name_mat['imgs']
        labels = label_mat['OBSCsn']
        labels = np.array(labels).astype(np.float32)
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'dsts', imgname[0][item][0]), labels[item][:]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class CuzGFolder(data.Dataset):

    def __init__(self, root, index, transform):
        name_mat = scio.loadmat("E:\CUZ2021\\imgsG.mat")
        label_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\OBSCsGn.mat")
        imgname = name_mat['imgsG']
        labels = label_mat['OBSCsGn']
        labels = np.array(labels).astype(np.float32)
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'dstsG', imgname[0][item][0]), labels[item][:]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class CuzPFolder(data.Dataset):

    def __init__(self, root, index, transform):
        name_mat = scio.loadmat("E:\CUZ2021\\imgsP.mat")
        label_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\OBSCsPn.mat")
        imgname = name_mat['imgsP']
        labels = label_mat['OBSCsPn']
        labels = np.array(labels).astype(np.float32)
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'dstsP', imgname[0][item][0]), labels[item][:]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class CuzEFolder(data.Dataset):

    def __init__(self, root, index, transform):
        name_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\imgsE.mat")
        label_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\OBSCsExtn.mat")
        imgname = name_mat['imgsE']
        labels = label_mat['OBSCsExtn']
        labels = np.array(labels).astype(np.float32)
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'distorted_images', imgname[0][item][0]), labels[item][:]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class CuzEGFolder(data.Dataset):

    def __init__(self, root, index, transform):
        name_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\imgsEG.mat")
        label_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\OBSCsExtGn.mat")
        imgname = name_mat['imgsEG']
        labels = label_mat['OBSCsExtGn']
        labels = np.array(labels).astype(np.float32)
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'distorted_images', imgname[0][item][0]), labels[item][:]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class CuzEPFolder(data.Dataset):

    def __init__(self, root, index, transform):
        name_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\imgsEP.mat")
        label_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\OBSCsExtPn.mat")
        imgname = name_mat['imgsEP']
        labels = label_mat['OBSCsExtPn']
        labels = np.array(labels).astype(np.float32)
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'distorted_images', imgname[0][item][0]), labels[item][:]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class KadidFolder(data.Dataset):

    def __init__(self, root, index, transform):
        name_mat = scio.loadmat("H:\kadis700k\kadis700k\\dstL.mat")
        label_mat = scio.loadmat("H:\kadis700k\kadis700k\\lbs.mat")
        imgname = name_mat['dstL']
        labels = label_mat['lbs']
        labels = np.array(labels).astype(np.float32)
        mlabels = np.mean(labels, axis=1)
        stdlables = np.std(labels, axis=1)
        stdlables = 1 - (stdlables / 0.39)
        sample = []
        for i, item in enumerate(index):
            for didx in range(10):
                sample.append((os.path.join(root, 'dist_imgs', imgname[item * 10 + didx][0][0]),
                               labels[item * 10 + didx][:], mlabels[item * 10 + didx], stdlables[item * 10 + didx]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, mm, ss = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target, mm, ss

    def __len__(self):
        length = len(self.samples)
        return length


class TID2013Folder(data.Dataset):

    def __init__(self, root, index, transform):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath, '.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        labels = (labels-np.min(labels)) / (np.max(labels)-np.min(labels))
        refnames_all = np.array(refnames_all)

        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
