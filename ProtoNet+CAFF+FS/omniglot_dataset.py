# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}


class OmniglotDataset(data.Dataset):
    vinalys_baseurl = 'E:/dataset/splits/'
    vinyals_split_sizes = {
        'test': vinalys_baseurl + 'test.txt',
        'train': vinalys_baseurl + 'train.txt',
        'trainval': vinalys_baseurl + 'trainval.txt',
        'val': vinalys_baseurl + 'val.txt',
    }

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_folder = 'splits/'
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, mode='train', root='..' + os.sep + 'dataset', transform=None, target_transform=None, download=True):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(OmniglotDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # if download:
        #     self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')
        self.classes = get_current_classes(os.path.join(
            self.root, self.splits_folder, mode + '.txt'))
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder), self.classes)

        self.idx_classes = index_classes(self.all_items)

        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))])

        self.x = map(self.load_img, paths, range(len(paths)))
        self.x = list(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]

        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        filename = self.all_items[index][0]
        # rot = self.all_items[index][-1]
        img = str.join(os.sep, [self.all_items[index][2], filename])
        target = self.idx_classes[self.all_items[index][1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))

    # def download(self):
    #     from six.moves import urllib
    #     import zipfile
    #
    #     if self._check_exists():
    #         return
    #
    #     try:
    #         os.makedirs(os.path.join(self.root, self.splits_folder))
    #         os.makedirs(os.path.join(self.root, self.raw_folder))
    #         os.makedirs(os.path.join(self.root, self.processed_folder))
    #     except OSError as e:
    #         if e.errno == errno.EEXIST:
    #             pass
    #         else:
    #             raise
    #
    #     for k, url in self.vinyals_split_sizes.items():
    #         print('== Downloading ' + url)
    #         data = urllib.request.urlopen(url)
    #         filename = url.rpartition('/')[-1]
    #         file_path = os.path.join(self.root, self.splits_folder, filename)
    #         with open(file_path, 'wb') as f:
    #             f.write(data.read())
    #
    #     for url in self.urls:
    #         print('== Downloading ' + url)
    #         data = urllib.request.urlopen(url)
    #         filename = url.rpartition('/')[2]
    #         file_path = os.path.join(self.root, self.raw_folder, filename)
    #         with open(file_path, 'wb') as f:
    #             f.write(data.read())
    #         orig_root = os.path.join(self.root, self.raw_folder)
    #         print("== Unzip from " + file_path + " to " + orig_root)
    #         zip_ref = zipfile.ZipFile(file_path, 'r')
    #         zip_ref.extractall(orig_root)
    #         zip_ref.close()
    #     file_processed = os.path.join(self.root, self.processed_folder)
    #     for p in ['images_background', 'images_evaluation']:
    #         for f in os.listdir(os.path.join(orig_root, p)):
    #             shutil.move(os.path.join(orig_root, p, f), file_processed)
    #         os.rmdir(os.path.join(orig_root, p))
    #     print("Download finished.")
    def load_img(self,path, idx):
        # path, rot = path.split(os.sep + 'rot')
        # if path in IMG_CACHE:
        #     x = IMG_CACHE[path]
        # else:
        #     x = Image.open(path)
        #
        #     IMG_CACHE[path] = x
        x = Image.open(path)
        # x.show()
        id=path[-5]
        per=int(id)-2
        sing_path=path[:-5]+'1'+path[-4:]
        x_single = Image.open(sing_path)
        x_single=x_single.rotate(float(per*15))
        # x_single.show()
        if self.transform:
            x = self.transform(x)
            x_single = self.transform(x_single)

        x = torch.cat([x,x_single],dim=0)
        # x = x.rotate(float(rot))
        # x = x.resize((28, 28))

        # shape = 3, x.size[0], x.size[1]
        # x = np.array(x, np.float32, copy=False)
        # x = 1.0 - torch.from_numpy(x)
        # x = x.transpose(0, 1).contiguous().view(shape)

        return x


def find_items(root_dir, classes):
    retour = []
    # rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            r = root.split(os.sep)
            lr = len(r)
            label = r[lr - 1]
            # for rot in rots:
            if label+os.sep in classes and (f.endswith("jpg")):
                if not f.endswith("1.jpg"):
                    retour.extend([(f, label, root)])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    return classes



