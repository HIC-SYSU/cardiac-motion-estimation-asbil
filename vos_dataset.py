import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import nibabel



class data_augmentation():

    def RandomFlip(self, images, labels=None):
        
        if random.random() < 0.3:
            images = torch.flipud(images)
            labels = torch.flipud(labels) if labels is not None else None
        elif random.random() > 0.7:
            images = torch.fliplr(images)
            labels = torch.fliplr(labels) if labels is not None else None

        if labels is not None:
            return images, labels 
        else:
            return images
    
    def RandomRotate(self, images, labels=None):

        if random.random() < 0.7:
            random_rotation = transforms.RandomRotation(degrees=(-90, 90))
            images = random_rotation(images)
            labels = random_rotation(labels) if labels is not None else None

        if labels is not None:
            return images, labels 
        else:
            return images

class VOSDataset(Dataset):


    def __init__(self, augmentation=False):
        

        self.augmentation = augmentation 
        self.cases = []
        self.note = {}
        self.num = 0

        self.Img_pth = [r'/data/images'] 
        self.lbs_pth = [r'/data/labels']
        self.patients = []
        for pth in self.lbs_pth:
            self.patients += sorted(os.listdir(pth))
            length = len(self.patients)  
            patients = self.patients

        for patient in patients:
            for pth in self.lbs_pth:
                if not os.path.exists(os.path.join(pth, patient)):
                    continue
                slc_seqs = sorted(os.listdir(os.path.join(pth, patient)))
                for frame in slc_seqs:
                    self.cases.append(os.path.join(patient, frame))

        if self.augmentation:
            self.D = data_augmentation()
            self.transform = [self.D.RandomFlip, 
                              self.D.RandomRotate, 
                            ]
        print(f'{len(self.cases)} 2D cardiac cases.')

    def normalize_data(self, img_np):

        cmin = torch.min(img_np)
        cmax = torch.max(img_np)
        img_np = (img_np - cmin) / (cmax- cmin + 0.0001)

        return img_np

    def split_masks(self, masks):
        unique = [x for x in torch.unique(masks) if x != 0]
        Masks = []
        for u in unique:
            mask = torch.zeros_like(masks)
            mask[masks == u] = 1
            Masks.append(mask)
        return Masks

    def __getitem__(self, idx):

        data = self.get_data(idx)

        return data

    def __len__(self):

        return len(self.cases)
    
    def get_data(self,idx):

        Case = self.cases[idx]
        img, gt = self.load_data(Case)
        img = self.normalize_data(img)
        if self.augmentation:
            for m in self.transform:
                img, gt = m(img, gt)

        patient = Case.split('/')[0]
        slices = Case.split('/')[1]

        lv, myo = self.split_masks(gt)
        
        data = {
                'images': img.unsqueeze(dim=1),
                'gts': myo,
                'split_gts': [myo, lv],
                'info': [patient, slices],
                }
        
        return data
        
    
    def load_data(self, case):
        ''' 
        case: patient/frame.nii.gz
        return: img, gt
        '''
        if isinstance(case, list):
            Data = []
            for c in case:
                pth = os.path.join(self.Img_pth[0], c)
                img = nibabel.load(pth).get_fdata()
                img = torch.Tensor(img)
                Data.append(img)
            return Data
        else:
            Img_pth = None
            Lbs_pth = None

            for pth in self.Img_pth:
                Img_pth = os.path.join(pth, case)
                if os.path.exists(Img_pth):
                    break
            img = nibabel.load(Img_pth).get_fdata()
            img = torch.Tensor(img)

            for pth in self.lbs_pth:
                Lbs_pth = os.path.join(pth, case)
                if os.path.exists(Lbs_pth):
                    break
            gt = nibabel.load(Lbs_pth).get_fdata() if Lbs_pth is not None else None
            gt = torch.Tensor(gt).long() if gt is not None else torch.zeros_like(img).long()

            return img, gt




