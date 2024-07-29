import os
import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets
from SimpleITK import Mask
from monai.transforms import Compose,SpatialPad
from monai.data import ImageDataset,DataLoader
import torch
from torch import tensor
from torch.utils.data import WeightedRandomSampler
import nibabel as nib
import numpy as np
import pandas as pd
import os
from monai.transforms import (
    EnsureChannelFirst,
    RandZoom,
    Compose,
    RandRotate,
    RandFlip,
    RandGaussianNoise,
    ToTensor,
    Resize,
    Rand3DElastic,
    RandSpatialCrop,
    ScaleIntensityRange,
    CenterSpatialCrop,
    Resize,
    NormalizeIntensity,
    ResizeWithPadOrCrop,
    SpatialPad,
    RandSpatialCrop
    )
from torch.utils.data import WeightedRandomSampler
import SimpleITK as sitk

class Balanced_sampler(WeightedRandomSampler):
    def __init__(self,labels:tensor,num_class=2,*args,**kwargs) -> None:
        """
        args:
            labels: torch tensor 
        """
        labels = np.asarray(labels).astype(int)
        class_freq = [len(np.where(labels==i)[0]) for i in range(num_class)]
        weights = [1.0/class_freq[label] for label in labels]
        #print(len(class_freq),len(weights))
        super().__init__(weights=weights,num_samples=len(weights),replacement=True,*args,**kwargs)

def get_loader(data, data_path,model='Res',
               train_data_label = None,
               test_data_label = None,
               label_name=None,
               ):
    # dataset normalize values
    if data == 'cifar100':
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
    elif data == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        stdv = [0.247, 0.243, 0.262]
    elif data == 'svhn':
        mean = [0.5, 0.5, 0.5]
        stdv = [0.5, 0.5, 0.5]


    #transforms
    print(model,'type of model')
    if model == 'Res':
        train_transforms = Compose([
            EnsureChannelFirst(),
            RandZoom(prob=0.5, min_zoom=1.0, max_zoom=1.2),
            RandRotate(range_z=0.35, prob=0.8),
            RandFlip(prob=0.5),
            NormalizeIntensity(),
            ToTensor()
        ])

        test_transforms = Compose([
            EnsureChannelFirst(),
            NormalizeIntensity(),
            ToTensor()
        ])
    elif model.startswith('Swin'):
        train_transforms = {
            'EnsureChannelFirst':EnsureChannelFirst(),
            #'Resize':Resize(cfg.Augmentation.Resize),
             'SpatialPad':SpatialPad((256,256,32)),
            'RandSpatialCrop':CenterSpatialCrop((256,256,32)),
            #'RandSpatialCrop':RandSpatialCrop(cfg.RandSpatialCrop,random_size=False,random_center=False),
            'RandZoom':RandZoom(prob=0.5, min_zoom=1.0, max_zoom=1.2),
            'RandRotate':RandRotate(range_z=0.35,prob=0.8),
            'RandFlip':RandFlip(prob=0.5),
            'NormalizeIntensity':NormalizeIntensity(),
            'ToTensor':ToTensor(),
        }

        test_transforms = {
            'EnsureChannelFirst':EnsureChannelFirst(),
            #'Resize':Resize(cfg.Augmentation.Resize),
            'SpatialPad':SpatialPad((256,256,32)),
            'CenterSpatialCrop':CenterSpatialCrop((256,256,32)),
            'NormalizeIntensity':NormalizeIntensity(),
            'ToTensor':ToTensor(),
        }
    # augmentation
    # train_transforms = tv.transforms.Compose([
    #     tv.transforms.RandomCrop(32, padding=4),
    #     tv.transforms.RandomHorizontalFlip(),
    #     tv.transforms.ToTensor(),
    #     tv.transforms.Normalize(mean=mean, std=stdv),
    # ])

    # test_transforms = tv.transforms.Compose([
    #     tv.transforms.ToTensor(),
    #     tv.transforms.Normalize(mean=mean, std=stdv),
    # ])

    # load datasets
    if data == 'cifar100':
        train_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                      train=True,
                                      transform=train_transforms,
                                      download=True)
        test_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=False,
                                     transform=test_transforms,
                                     download=False)
    elif data == 'cifar10':  # cifar10_data /cifiar10_data
        train_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                     train=True,
                                     transform=train_transforms,
                                     download=True)
        test_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                    train=False,
                                    transform=test_transforms,
                                    download=False)
    elif data == 'svhn':
        train_set = datasets.SVHN(root=os.path.join(data_path, 'svhn_data'),
                                  split='train',
                                  transform=train_transforms,
                                  download=True)
        test_set = datasets.SVHN(root=os.path.join(data_path, 'svhn_data'),
                                 split='test',
                                 transform=test_transforms,
                                 download=True)
    elif data == 'HGP':
        train_data = DataFiles(data_path,train_data_label,label_name)
        test_data = DataFiles(data_path,test_data_label,label_name)
        train_images = sorted(train_data.get_images())
        train_labels = train_data.get_labels()
  
        vali_images = sorted(test_data.get_images())
        vali_labels = test_data.get_labels()
        train_data.Data_check()
        test_data.Data_check()
        train_data = CreateImageDataset(image_files=train_images,labels=train_labels,transform_methods=train_transforms)
        test_data = CreateImageDataset(image_files=vali_images,labels=vali_labels,transform_methods=test_transforms)

    # make Custom_Dataset
    if data == 'HGP':
        test_onehot = one_hot_encoding(test_data.labels)
        test_label = test_data.labels
    else:
        train_data = Custom_Dataset(train_set.data,
                                    train_set.targets,
                                    'cifar', train_transforms)
        test_data = Custom_Dataset(test_set.data,
                                   test_set.targets,
                                   'cifar', test_transforms)
        # one_hot_encoding
        test_onehot = one_hot_encoding(test_set.targets)
        test_label = test_set.targets
    
    sampler = Balanced_sampler(train_labels,num_class=2)
    #sampler = None
    # make DataLoader
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                            sampler = sampler,
    #                                            num_workers=0)
    # test_loader = torch.utils.data.DataLoader(test_data,
    #                                           shuffle=True,
    #                                           num_workers=0)
    
    train_loader = CreateDataLoader(dataset=train_data,num_workers=0,sampler=sampler,batch_size=1).build_train_loader() 
    test_loader = CreateDataLoader(dataset=test_data,num_workers=0,batch_size=1).build_vali_loader()
    print("-------------------Make loader-------------------")
    print('Train Dataset :',len(train_loader.dataset),
          '   Test Dataset :',len(test_loader.dataset))

    return train_loader, test_loader, test_onehot, test_label







class DataFiles:
    def __init__(self,
                 data_path: str,
                 label_path: str,
                 label_name: str) -> None:
        """
        List all the files in a data directory, using csv file to get the labels and their paths.
        Args:
            data_path: csv path to the data
            label_path: path to the csv file containing the labels
            label_name: name of the predicted label
        """
        self.data_path = data_path
        self.label_path = label_path
        self.label_name = label_name

    def get_images(self):
        filenames = self.get_data_path()
        return [os.path.join(self.data_path, filename) for filename in filenames]

    def get_masks(self):
        filenames = self.get_mask_path()
        return [os.path.join(self.data_path, filename) for filename in filenames]

    def get_labels(self):
        return pd.read_csv(self.label_path)[self.label_name].values.tolist()
    
    def get_data_path(self):
        return pd.read_csv(self.label_path)['data_path'].values.tolist()
    
    def get_mask_path(self):
        return pd.read_csv(self.label_path)['mask_path'].values.tolist()

    def Data_check(self):
        assert len(self.get_images()) == len(self.get_labels()) , 'The number of images and labels are not equal'

    def generate_data_dic(self):
        pass



class CreateImageDataset(ImageDataset):
    def __init__(self,
                 image_files: list,
                 labels: list,
                 transform_methods=None,
                 *args,
                 **kwargs):
        """
        Args:
            image_files: list of image files
            labels: list of labels
            transform_methods: list of transform methods
            data_aug: bool, whether to do data augmentation
            padding_size: tuple, the size of padding. For models that require fixed size
        """

        self.train_transform = transform_methods

        super().__init__(image_files=image_files,
                            labels=labels,
                            transform=transform_methods,
                         *args,
                           **kwargs)
        print(labels,'labels!')
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,index,*args,**kwargs):
        img = self.loader(self.image_files[index])
        img = self.transform(img)
        print(img.shape,'img shape')

        label = self.labels[index]
        return img, label , index


        


class CreateDataLoader(DataLoader):
    def __init__(self,dataset,batch_size,num_workers=0,*args,**kwargs):
        super().__init__(dataset=dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False,*args,**kwargs)
        self.args = args
        self.kwargs = kwargs
    
    def build_train_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,num_workers=self.num_workers,drop_last=True,*self.args,**self.kwargs)

    def build_vali_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)
    
    def build_test_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)
# Custom_Dataset class
class Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, transform=None):
        self.x_data = x
        self.y_data = y
        self.data = data_set
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    # return idx
    def __getitem__(self, idx):
        if self.data == 'cifar':
            img = Image.fromarray(self.x_data[idx])
        elif self.data == 'svhn':
            img = Image.fromarray(np.transpose(self.x_data[idx], (1, 2, 0)))

        x = self.transform(img)

        return x, self.y_data[idx], idx

def one_hot_encoding(label):
    print("one_hot_encoding process")
    cls = set(label)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, label)))

    return one_hot

if __name__ == '__main__':
    data_path = '../Data/Mixed_HGP/'
    train_data_label = '../Data/Mixed_HGP_Folds/train_cv_0.csv'
    test_data_label = '../Data/Mixed_HGP_Folds/val_cv_0.csv'
    label_name = 'HGP_Type'
    train_loader, test_loader, test_onehot, test_label = get_loader(data='HGP', data_path=data_path,train_data_label=train_data_label,test_data_label=test_data_label,label_name=label_name)
    for i, (input, target, idx) in enumerate(train_loader):
       print(i, input.size(), target, idx)
    
    for i, (input, target, idx) in enumerate(train_loader):
       print(i, input.size(), target, idx)