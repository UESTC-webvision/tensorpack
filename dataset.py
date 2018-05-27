from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.utils import logger
from tensorpack.utils.loadcaffe import get_caffe_pb
from tensorpack.utils.fs import mkdir_p, download, get_dataset_path
from tensorpack.utils.timer import timed_operation
import os
import numpy as np
import cv2

VALID_PATH = './val_images_resized/'

class ILSVRCMeta(object):
    """
    Provide methods to access metadata for ILSVRC dataset.
    """

    def __init__(self, dir=None):
        self.dir = dir
        mkdir_p(self.dir)
        f = os.path.join(self.dir, 'synsets.txt')
        self.caffepb = None

    def get_synset_words_1000(self):
        """
        Returns:
            dict: {cls_number: cls_name}
        """
        fname = os.path.join(self.dir, 'synset_words.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_synset_1000(self):
        """
        Returns:
            dict: {cls_number: synset_id}
        """
        fname = os.path.join(self.dir, 'synsets.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_image_list(self, name, dir_structure='original'):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
            dir_structure (str): same as in :meth:`ILSVRC12.__init__()`.
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']
        assert dir_structure in ['original', 'train']
        add_label_to_fname = (name != 'train' and dir_structure != 'original')
        if add_label_to_fname:
            synset = self.get_synset_1000()

        fname = os.path.join(self.dir, name + '.txt')
        assert os.path.isfile(fname), fname
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                cls = int(cls)

                if add_label_to_fname:
                    name = os.path.join(synset[cls], name)

                ret.append((name.strip(), cls))
        assert len(ret), fname
        return ret

    def get_per_pixel_mean(self, size=None):
        """
        Args:
            size (tuple): image size in (h, w). Defaults to (256, 256).
        Returns:
            np.ndarray: per-pixel mean of shape (h, w, 3 (BGR)) in range [0, 255].
        """
        if self.caffepb is None:
            self.caffepb = get_caffe_pb()
        obj = self.caffepb.BlobProto()

        mean_file = os.path.join(self.dir, 'imagenet_mean.binaryproto')
        with open(mean_file, 'rb') as f:
            obj.ParseFromString(f.read())
        arr = np.array(obj.data).reshape((3, 256, 256)).astype('float32')
        arr = np.transpose(arr, [1, 2, 0])
        if size is not None:
            arr = cv2.resize(arr, size[::-1])
        return arr


def _guess_dir_structure(dir):
    subdir = os.listdir(dir)[0]
    # find a subdir starting with 'n'
    if subdir.startswith('n') and \
            os.path.isdir(os.path.join(dir, subdir)):
        dir_structure = 'train'
    else:
        dir_structure = 'original'
    logger.info(
        "[ILSVRC12] Assuming directory {} has '{}' structure.".format(
            dir, dir_structure))
    return dir_structure

def read_filelist (data_path): #读取filelist
    file_list = open(data_path, 'r')
    imgs = []
    for line in file_list:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        imgs.append((words[0], int(words[1])))
    return imgs

class webvisionFiles(RNGDataFlow):
    def __init__(self, dir, name, shuffle=None):
        """
        Same as in :class:`ILSVRC12`.
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = './' if name == 'train' else VALID_PATH
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        self.imglist = read_filelist(dir)

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            fname = os.path.join(self.full_dir, fname)
            yield [fname, label]

class webvision(webvisionFiles):
    """
    Produces uint8 ILSVRC12 images of shape [h, w, 3(BGR)], and a label between [0, 999].
    """
    def __init__(self, dir, name, shuffle=None):
        super(webvision, self).__init__(dir, name, shuffle)


    """
    There are some CMYK / png images, but cv2 seems robust to them.
    https://github.com/tensorflow/models/blob/c0cd713f59cfe44fa049b3120c417cc4079c17e3/research/inception/inception/data/build_imagenet_data.py#L264-L300
    """
    def get_data(self):
        for fname, label in super(webvision, self).get_data():
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            yield [im, label]