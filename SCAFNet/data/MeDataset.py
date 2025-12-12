import torchvision.transforms
import glob
from torch.utils.data.dataset import Dataset
from data.util import *
import torch


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


def rgb2ycbcr(image):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    assert image.size(1) == 3, "输入图像的通道数必须为3"

    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = - 0.168736 * r - 0.331264 * g + 0.5 * b + 128 / 255.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128 / 255.0
    return y, cb, cr


def ycbcr2rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128 / 255.0)
    g = y - 0.34414 * (cb - 128 / 255.0) - 0.71414 * (cr - 128 / 255.0)
    b = y + 1.772 * (cb - 128 / 255.0)
    return torch.cat([r, g, b], dim=1)


class FusionDataset(Dataset):
    def __init__(self,
                 split,
                 min_max=(-1, 1),
                 mri_path='dataset/PET-MRI/train/MRI',
                 pet_path='dataset/PET-MRI/train/PET'):
        super(FusionDataset, self).__init__()
        assert split in ['train', 'val',
                         'test'], 'split must be "train"|"val"|"test"'
        self.min_max = min_max
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.1)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.1)

        data_dir_vis = pet_path
        data_dir_ir = mri_path
        self.filepath_pet, self.filenames_pet = prepare_data_path(data_dir_vis)
        self.filepath_mri, self.filenames_mri = prepare_data_path(data_dir_ir)
        self.split = split
        self.length = min(len(self.filenames_pet), len(self.filenames_mri))

    def __getitem__(self, index):
        if self.split == 'train':
            pet_image = Image.open(self.filepath_pet[index])
            mri_image = Image.open(self.filepath_mri[index])
            # Random horizontal flipping
            if random.random() > 0.5:
                pet_image = self.hflip(pet_image)
                mri_image = self.hflip(mri_image)
            # Random vertical flipping
            if random.random() > 0.5:
                pet_image = self.vflip(pet_image)
                mri_image = self.vflip(mri_image)

            pet_image = ToTensor()(pet_image) * \
                (self.min_max[1] - self.min_max[0]) + self.min_max[0]
            mri_image = ToTensor()(mri_image) * \
                (self.min_max[1] - self.min_max[0]) + self.min_max[0]
            # if pet_image.size(0) == 3:
            # pet_image, cb, cr = rgb2ycbcr(pet_image)
            # return {'pet_y': pet_image, 'mri': mri_image, 'cb': cb, 'cr': cr}, self.filenames_mri[index]
            # else:
            return {'pet_y': pet_image, 'mri': mri_image,}, self.filenames_mri[index]

        elif self.split == 'test':
            pet_image = Image.open(self.filepath_pet[index])
            mri_image = Image.open(self.filepath_mri[index])
            pet_image = ToTensor()(pet_image)
            pet_image = pet_image * \
                (self.min_max[1] - self.min_max[0]) + self.min_max[0]

            mri_image = ToTensor()(mri_image)
            mri_image = mri_image * \
                (self.min_max[1] - self.min_max[0]) + self.min_max[0]
            # if pet_image.size(0) == 3:
            # pet_image, cb, cr = rgb2ycbcr(pet_image)
            # return {'pet_y': pet_image, 'mri': mri_image, 'cb': cb, 'cr': cr}, self.filenames_mri[index]
            # else:
            return {'pet_y': pet_image, 'mri': mri_image}, self.filenames_mri[index]

    def __len__(self):
        return self.length
