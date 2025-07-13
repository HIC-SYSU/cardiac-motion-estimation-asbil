import os
from os import path
import random
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
# from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
# import skimage.exposure as exp
import nibabel
import json
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.cuda.device_count.cache_clear()
def savenii(input, pth):
    if isinstance(input, list):
        input = np.stack(input, axis=0)
    header = nib.Nifti1Header()
    new_image = nib.Nifti1Image(input, np.eye(4), header=header) 
    nib.save(new_image, pth) 

def subplots(input):
    input = input.squeeze()
    if isinstance(input, torch.Tensor):
        if input.is_cuda:
            input = input.detach().cpu()
    length = input.squeeze().shape[0]
    num = math.ceil(math.sqrt(length))
    _, axe = plt.subplots(num,num)
    for i in range(num):
        for j in range(num):
            if i+num*j >= length:
                continue
            axe[i][j].imshow(input[i+num*j], cmap='gray')
    plt.show()

def save_json(input,save):
    with open(save, "w") as file:
        json.dump(input, file)

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

    def RandomResizedCrop(self, images, labels=None):
        
        if random.random() < 0.5:
            t,h,w = images.shape
            random_resized_crop_IM = transforms.RandomResizedCrop((h, w), interpolation = InterpolationMode.BILINEAR)
            images = random_resized_crop_IM(images)
            if labels is not None:
                random_resized_crop_LB = transforms.RandomResizedCrop((h, w), interpolation = InterpolationMode.NEAREST)
                labels = random_resized_crop_LB(labels)

        if labels is not None:
            return images, labels 
        else:
            return images

    def RandomEqualize(self, images, labels=None):
        
        if random.random() < 0.5:
            equalizer = transforms.RandomEqualize()
            images = equalizer(images)

        if labels is not None:
            return images, labels 
        else:
            return images

    def RandomContrastAdjustment(self, images, labels=None):
        
        brightness_factor = random.uniform(0.1, 3.8)
        images = images + (brightness_factor - 1) * 0.5
        images = torch.clamp(images, -1, 1) 
        if labels is not None:
            return images, labels
        else:
            return images

    def RandomGaussianBlur(self, images, labels=None):

        if random.random() < 0.5:
            intensity_var = 1 + torch.clip(np.random.normal(), -0.5, 0) * 0.1
            images = images * intensity_var

        if labels is not None:
            return images, labels
        else:
            return images

class VOSDataset(Dataset):
    '''
    name: anzhen, shengyi, tag, camus
    mode: train, val
    dim: 2D, 3D
    view: A2C, A4C
    augmentation: True, False
    '''

    def __init__(self, name, mode, dim, view = 'A2C', augmentation=False, interval = None):
        
        self.name = name
        self.mode = mode
        self.dim = dim
        self.view = view
        self.interval = interval
        self.augmentation = augmentation if self.mode == 'train' else False
        self.cases = []
        self.note = {}
        self.num = 0

        if self.name == "anzhen":
            self.Img_pth = [os.path.join(r'/data/zhuangshuxin/datasets/2D_cineMRI_anzhen', 'videos'),
                            # r'/data/zhuangshuxin/datasets/anzhen_CRT_cine/images'
                            ]
            self.lbs_pth = [os.path.join(r'/data/zhuangshuxin/datasets/2D_cineMRI_anzhen', 'annotations'),
                            # r'/data/zhuangshuxin/datasets/anzhen_CRT_cine/labelsx'
                            ]  
            self.resize_transformIM = transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR)
            self.resize_transformLB = transforms.Resize((128, 128), interpolation=InterpolationMode.NEAREST)

        if self.name == "crt":
            self.Img_pth = [r'/data/zhuangshuxin/datasets/anzhen_CRT_cine/images'
                            ]
            self.lbs_pth = [r'/data/zhuangshuxin/datasets/anzhen_CRT_cine/labelsx'
                            ]  
            self.resize_transformIM = transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR)
            self.resize_transformLB = transforms.Resize((128, 128), interpolation=InterpolationMode.NEAREST)
            
        elif self.name == "shengyi":
            self.Img_pth = [r'/data/zhuangshuxin/datasets/shengyi_cmr_1/crop_CMR'] 
            self.lbs_pth = [r'/data/zhuangshuxin/datasets/shengyi_cmr_1/crop_labels_x']
            self.resize_transformIM = transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR)
            self.resize_transformLB = transforms.Resize((128, 128), interpolation=InterpolationMode.NEAREST)
        
        elif self.name == "tag":
            self.Img_pth = [r'/data/zhuangshuxin/datasets/shengyi_tmr/crop-videos'] 
            self.lbs_pth = [r'/data/zhuangshuxin/datasets/shengyi_tmr/crop-landmarks']
            self.resize_transformIM = transforms.Resize((160, 160), interpolation=InterpolationMode.BILINEAR)
            self.resize_transformLB = transforms.Resize((160, 160), interpolation=InterpolationMode.NEAREST)
            
        elif self.name == "camus":
            self.Img_pth = [os.path.join(r'/data/zhuangshuxin/datasets/CAMUS', self.view, 'videos')]
            self.lbs_pth = [os.path.join(r'/data/zhuangshuxin/datasets/CAMUS', self.view, 'annotations')]
            self.resize_transformIM = transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)
            self.resize_transformLB = transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)
            self.dim = '2D'

        elif self.name == "ACDC":
            self.Img_pth = [f'/data/zhuangshuxin/datasets/ACDC/{self.mode}']
            self.lbs_pth = [f'/data/zhuangshuxin/datasets/ACDC/{self.mode}']
            self.resize_transformIM = transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR)
            self.resize_transformLB = transforms.Resize((128, 128), interpolation=InterpolationMode.NEAREST)
            self.dim = '4D'
            self.json = json.load(open(f'/data/zhuangshuxin/datasets/ACDC/{self.mode}.json'))

        elif self.name == "all":
            self.Img_pth = [r'/data/zhuangshuxin/datasets/shengyi_cmr_1/crop_CMR', 
                            os.path.join(r'/data/zhuangshuxin/datasets/2D_cineMRI_anzhen', 'videos'),
                            r'/data/zhuangshuxin/datasets/anzhen_CRT_cine/images'
                            ]
            self.lbs_pth = [r'/data/zhuangshuxin/datasets/shengyi_cmr_1/crop_labels_x', 
                            os.path.join(r'/data/zhuangshuxin/datasets/2D_cineMRI_anzhen', 'annotations'),
                            r'/data/zhuangshuxin/datasets/anzhen_CRT_cine/labelsx'
                            ]
            self.resize_transformIM = transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR)
            self.resize_transformLB = transforms.Resize((128, 128), interpolation=InterpolationMode.NEAREST)

        elif self.name == "MM":
            self.Img_pth = [f'/data/zhuangshuxin/datasets/M&M/{self.mode}']
            self.lbs_pth = [f'/data/zhuangshuxin/datasets/M&M/{self.mode}']
            self.resize_transformIM = transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR)
            self.resize_transformLB = transforms.Resize((128, 128), interpolation=InterpolationMode.NEAREST)
            self.dim = '4D'

            self.text = pd.read_csv(r'/data/zhuangshuxin/datasets/M&M/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
            self.json = json.load(open(f'/data/zhuangshuxin/datasets/M&M/{self.mode}.json'))
            names = self.text.values[:,1]
            EDES = np.array([self.text.values[:,5], self.text.values[:,6]]).transpose(1,0)
            self.Dicts = dict(zip(names, EDES))
            
        self.patients = []
        for pth in self.lbs_pth:
            self.patients += sorted(os.listdir(pth))
        if self.name not in ['all', "ACDC", "MM"]:
            length = len(self.patients)  
            patients = self.patients[:int(length*0.8)] if self.mode == 'train' else self.patients[int(length*0.8):]
        else:
            patients = self.patients

        # counting based on sequences
        if self.dim == '2D':
            for patient in patients:
                for pth in self.lbs_pth:
                    if not os.path.exists(os.path.join(pth, patient)):
                        continue
                    if self.name not in ['camus']:
                        slc_seqs = sorted(os.listdir(os.path.join(pth, patient)))
                        for frame in slc_seqs:
                            self.cases.append(os.path.join(patient, frame))
                    else:
                        self.cases.append(os.path.join(patient))

            print('%d 2D cardiac cineMR cases.' % (len(self.cases)))

        # counting based on patients
        elif self.dim == '3D':
            self.cases = {patient: None for patient in patients}
            for patient in patients:
                for pth in self.lbs_pth:
                    if not os.path.exists(os.path.join(pth, patient)):
                        continue
                    self.cases[patient] = sorted(os.listdir(os.path.join(pth, patient)))

            print('%d 3D cardiac cineMR cases.' % (len(self.cases)))
        
        elif self.dim == '4D':
            for patient in self.patients:
                for pth in self.lbs_pth:
                    self.cases.append(patient)
            self.cases = sorted(self.cases) 

        if self.augmentation:
            self.D = data_augmentation()
            self.transform = [self.D.RandomFlip, 
                              self.D.RandomRotate, 
                              self.D.RandomContrastAdjustment,
                            #   self.D.RandomResizedCrop, 
                            #   self.D.RandomEqualize,
                            ]
        
    def find_index(self, name, slc=None, dim='3D'):

        if name not in self.patients:
            raise ValueError('not in the datasets')
        else:
            impth = os.path.join(self.Img_pth[0], name)
            lbpth = os.path.join(self.lbs_pth[0], name)

        if self.name not in ['ACDC', 'MM']:
            if dim == '3D':
                slcs = os.listdir(lbpth)
                Imgs, Lbs, orginal_imgs = [], [], []
                for slc in slcs:
                    im_case = os.path.join(impth, slc)
                    lb_case = os.path.join(lbpth, slc)
                    img = self.resize_transformIM(nibabel.load(im_case).get_fdata())
                    lbs = self.resize_transformLB(nibabel.load(lb_case).get_fdata())
                    orginal_imgs.append(img.clone().unsqueeze(1))
                    Imgs.append(self.normalize_data(img).unsqueeze(1))
                    Lbs.append(lbs)

                Imgs = torch.stack(Imgs, dim=1)
                Lbs = torch.stack(Lbs, dim=1)
                orginal_imgs = torch.stack(orginal_imgs, dim=1)
                if self.name not in ['tag', 'camus']:
                    lv, myo = self.split_masks(Lbs)
                    Lbs = myo
                
                return Imgs, Lbs, orginal_imgs
            
            elif dim == '2D':
                if slc != 'noslice':
                    impth = os.path.join(impth, slc)
                    lbpth = os.path.join(lbpth, slc)

                orginal_imgs = self.resize_transformIM(nibabel.load(impth).get_fdata())
                Imgs = self.normalize_data(orginal_imgs).unsqueeze(1)
                Lbs = self.resize_transformLB(nibabel.load(lbpth).get_fdata())
                if self.name not in ['tag']:
                    lv, myo = self.split_masks(Lbs)
                    Lbs = myo
                else:
                    Lbs[Lbs!=0] = 1
                return Imgs, Lbs, orginal_imgs
            
        elif self.name in ['ACDC']:
            cases  = sorted(os.listdir(impth))
            idxs = [0, int(cases[3].split('.')[0].split('frame')[-1])-1]
            slc = self.json[name]

            f1st_gt = os.path.join(impth, cases[2])
            fxth_gt = os.path.join(impth, cases[4])
            im_case = os.path.join(impth, cases[0])
            f1st_gt, fxth_gt, img = self.load_data([f1st_gt, fxth_gt, im_case])
            img = img.permute(3,2,0,1)[idxs[0]:idxs[1], slc[0]:slc[-1]+1]
            img = self.normalize_data(img).unsqueeze(dim=2)
            f1st_gt = f1st_gt.permute(2,0,1)[slc[0]:slc[-1]+1]
            fxth_gt = fxth_gt.permute(2,0,1)[slc[0]:slc[-1]+1]
            f1st_gt[f1st_gt!=2] = 0
            fxth_gt[fxth_gt!=2] = 0
            
            return img, f1st_gt, fxth_gt
        
        elif self.name in ['MM']:
            cases  = sorted(impth)
            edes = sorted(list(self.Dicts[name]))
            slc = self.json[name]
            cases  = sorted(os.listdir(os.path.join(self.Img_pth[0], name)))
            data_4D = os.path.join(impth, cases[0])
            gts = os.path.join(impth, cases[1])

            data_4D, gts = self.load_data([data_4D, gts])
            data_4D = data_4D.permute(3,2,1,0)[edes[0]:edes[1],slc[0]:slc[-1]]
            data_4D = self.normalize_data(data_4D)
            data_4D = self.resize_transformIM(data_4D)
            gts = gts.permute(3,2,1,0)[:,slc[0]:slc[-1]]
            gts = self.resize_transformLB(gts)
            idx_min = np.min(edes) 
            idx_max = np.max(edes)
            frmed_gt = gts[idx_min]
            frmes_gt = gts[idx_max]
            frmed_gt[frmed_gt!=2] = 0
            frmes_gt[frmes_gt!=2] = 0

            return data_4D.unsqueeze(dim=2), frmed_gt, frmes_gt
      
    def normalize_data(self, img_np):

        cmin = torch.min(img_np)
        cmax = torch.max(img_np)
        img_np = (img_np - cmin) / (cmax- cmin + 0.0001)

        if self.name in ['tag', 'shengyi']:
            meanv = torch.mean(img_np, dim=(1,2))
            meanv = meanv.view(-1,1,1)
            img_np = img_np / (2 * meanv)
            img_np = torch.clamp(img_np, -1, 1)
        # savenii(img_np, r'/data/zhuangshuxin/Codes/ASBIL-new/references/Checkpoints/diffusemorph_lag/test.nii.gz')
            # for i in range(img_np.shape[0]):
            #     meanv = torch.mean(img_np[i])
            #     img_np[i] = img_np[i] / (2 * meanv)
            #     img_np[i] = np.clip(img_np[i], -1, 1)
        return img_np

    def split_masks(self, masks):
        unique = [x for x in torch.unique(masks) if x != 0]
        # if len(unique) != 2:
        #     subplots(masks)
        Masks = []
        for u in unique:
            mask = torch.zeros_like(masks)
            mask[masks == u] = 1
            Masks.append(mask)
        return Masks

    def __getitem__(self, idx):

        if self.dim == '2D':
            data = self.get_2D_data(idx)
        elif self.dim == '3D':
            data = self.get_3D_data(idx)
        elif self.dim == '4D':
            if self.name == "ACDC":
                data = self.get_ACDC_data(idx)
            elif self.name == "MM":
                data = self.get_MM_data(idx)

        return data

    def __len__(self):

        return len(self.cases)
    
    def get_2D_data(self,idx):

        Case = self.cases[idx]
        img, gt = self.load_data(Case)

        img = self.resize_transformIM(img)
        gt = self.resize_transformLB(gt)
        img = self.normalize_data(img)
        if self.augmentation:
            for m in self.transform:
                img, gt = m(img, gt)

        # get frame index
        Length = img.shape[0]
        if self.mode == 'train':
            if self.interval is not None:
                start_index = torch.randint(0, Length - self.interval + 1, (1,))
                img = self.randon_select(img, idx=start_index, subseq=self.interval)
                gt = self.randon_select(gt, idx=start_index, subseq=self.interval)

        if self.name not in ['camus']:
            patient = Case.split('/')[0]
            slices = Case.split('/')[1]
        else:
            patient = Case
            slices = 'noslice'

        if self.name not in ['tag']:
            gt_cls = self.split_masks(gt)
            if len(gt_cls) == 2:
                lv, myo = gt_cls
                split_gts = [myo, lv]
            else:
                split_gts = [gt]
            
        else:
            split_gts = self.split_masks(gt) # endo, epi
            gt[gt!=0] = 1
        
        data = {
                'images': img.unsqueeze(dim=1),
                'gts': gt,
                'split_gts': split_gts,
                'info': [patient, slices],
                }
        
        return data
        
    def get_3D_data(self,idx):

        Case = sorted(list(self.cases.keys()))[idx]
        patient = Case.split('/')[0]
        Imgs = []
        Lbs = [] 
        for slc in self.cases[Case]:
            case = os.path.join(Case, slc)
            img, gt = self.load_data(case)
            img = self.resize_transformIM(img)
            gt = self.resize_transformLB(gt)
            Imgs.append(img), Lbs.append(gt)
        Imgs = torch.stack(Imgs, dim=1)
        Imgs = self.normalize_data(Imgs)

        Lbs = torch.stack(Lbs, dim=1).squeeze() 

        t,s,h,w = Imgs.shape 
        if self.augmentation:
            Imgs = Imgs.view(t*s, h, w)
            Lbs = Lbs.view(t*s, h, w)
            for m in self.transform:
                Imgs, Lbs = m(Imgs, Lbs)
        Imgs = Imgs.view(t,s,1,h,w)
        Lbs = Lbs.view(t,s,h,w)

        # get frame index
        Length = Imgs.shape[0]
 
        if self.mode == 'train':
            # frames_idx = sorted(random.sample(list(range(1, Length)), int(Length*0.6))) + [0]
            frames_idx = list(range(0, Length))
            # if slc_n > sp:
            #     Rand = random.randint(0, slc_n-sp)
            #     slice_idx = range(Rand,(Rand+sp))
            # else:
            #     slice_idx = range(0, slc_n)  
        elif self.mode == 'val':
            frames_idx = list(range(0, Length))

        imt = []
        lbt = []
        for i in frames_idx:
            imt.append(Imgs[i])
            lbt.append(Lbs[i])
        imt = torch.stack(imt, dim=0)
        lbt = torch.stack(lbt, dim=0).squeeze()  

        lv, myo = self.split_masks(lbt)

        data = {
                'images': imt,
                'gts': lbt,
                'split_gts': [myo, lv],
                'info': [patient, frames_idx],
                }
        
        return data
    
    def get_ACDC_data(self,idx):
        '''
        3: myocardium            2: myocardium
        2: left ventricle   ==>  1: left ventricle
        1: left atrium
        '''
        Case = self.cases[idx]
        slc = self.json[Case]
        cases  = sorted(os.listdir(os.path.join(self.Img_pth[0], Case)))

        data_4D = os.path.join(Case, cases[0])
        frame_1st = os.path.join(Case, cases[1])
        frame_1st_gt = os.path.join(Case, cases[2])
        frame_xth = os.path.join(Case, cases[3])
        frame_xth_gt = os.path.join(Case, cases[4]) 

        Data = self.load_data([data_4D, frame_1st, frame_1st_gt, frame_xth, frame_xth_gt])
        for i in range(len(Data)):
            Data[i] = Data[i].permute(2,0,1)[slc[0]:slc[-1]] if len(Data[i].shape) == 3 else Data[i].permute(3,2,0,1)[:,slc[0]:slc[-1]]   # [t,s,h,w]
            Data[i] = self.resize_transformIM(Data[i])
            if len(torch.unique(Data[i])) > 4:
                Data[i] = self.normalize_data(Data[i])

        data_4D, frame_1st, frame_1st_gt, frame_xth, frame_xth_gt = Data
        # subplots(frame_xth_gt)
        if self.augmentation:
            t, s, h, w = data_4D.shape
            data_4D = data_4D.reshape(t*s, h, w)
            for m in self.transform:
                data_4D = m(data_4D)
            data_4D = data_4D.reshape(t,s,1,h,w)

        patient = Case.split('/')[0]
        idxs = [0, int(cases[3].split('.')[0].split('frame')[-1])-1]

        if self.mode == 'train':
            t,s,c,h,w = data_4D.shape
            frames_idx = sorted(random.sample(list(range(1, t)), 7)) + [0]
            data_4D = torch.stack([data_4D[i] for i in frames_idx], dim=0)
            data = {'images': [data_4D], 
                    'info': [patient, frames_idx]}   
        elif self.mode == 'val':
            la_1st, myo_1st, lv_1st = self.split_masks(frame_1st_gt)
            la_xth, myo_xth, lv_xth = self.split_masks(frame_xth_gt)
            # a = torch.unique(myo_1st)
            # b = torch.unique(myo_xth)
            # c = torch.unique(lv_1st)
            # d = torch.unique(lv_xth)
            # print(a,b,c,d)
            data = {'images': data_4D[idxs[0]:idxs[1]].unsqueeze(dim=2), 
                    'gts': [frame_1st_gt, frame_xth_gt],
                    'split_gts': [[myo_1st, lv_1st], [myo_xth, lv_xth]],
                    'info': [patient, idxs]}
        else:
            raise ValueError('Invalid mode')
        
        return data

    def get_MM_data(self,idx):

        Case = self.cases[idx]
        edes = sorted(list(self.Dicts[Case]))
        slc = self.json[Case]
        cases  = sorted(os.listdir(os.path.join(self.Img_pth[0], Case)))

        data_4D = os.path.join(Case, cases[0])
        gts = os.path.join(Case, cases[1])

        Data = self.load_data([data_4D, gts])
        for i in range(len(Data)):
            Data[i] = Data[i].permute(3,2,1,0)[:,slc[0]:slc[-1]]   # [t,s,h,w]
            if len(torch.unique(Data[i])) > 4:
                Data[i] = self.normalize_data(Data[i])

        data_4D, gts = Data

        data_4D = self.resize_transformIM(data_4D)
        gts = self.resize_transformLB(gts)
        # subplots(data_4D[:,0])

        data_4D = data_4D[edes[0]:edes[1]]
        frame_ed_gt = gts[edes[0]]
        frame_es_gt = gts[edes[1]]
        # subplots(frame_ed_gt)

        t, s, h, w = data_4D.shape
        if self.augmentation:
            data_4D = data_4D.reshape(t*s, h, w)
            for m in self.transform:
                data_4D = m(data_4D)
            data_4D = data_4D.reshape(t,s,1,h,w)
        else:
            data_4D = data_4D.reshape(t,s,1,h,w)

        if self.mode == 'train':
            t,s,c,h,w = data_4D.shape
            frames_idx = list(range(t))
            # frames_idx = sorted(random.sample(list(range(1, t)), 7)) + [0]
            # data_4D = torch.stack([data_4D[i] for i in frames_idx], dim=0)
            data = {'images': data_4D, 
                    'info': [Case, frames_idx]}   
            
        elif self.mode == 'val':
            lv_1st, myo_1st, la_1st = self.split_masks(frame_ed_gt)
            lv_xth, myo_xth, la_xth = self.split_masks(frame_es_gt)
            # discard LA
            frame_es_gt[frame_es_gt==3] = 0
            frame_ed_gt[frame_ed_gt==3] = 0
            # subplots(frame_es_gt)
            # a = torch.unique(myo_1st)
            # b = torch.unique(myo_xth)
            # c = torch.unique(lv_1st)
            # d = torch.unique(lv_xth)
            # print(a,b,c,d)
            data = {'images': data_4D, 
                    'gts': [frame_ed_gt, frame_es_gt],
                    'split_gts': [[myo_1st, lv_1st], [myo_xth, lv_xth]],
                    'info': [Case]}
        else:
            raise ValueError('Invalid mode')
        
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

    def randon_select(self, vid, idx, subseq=9):
        total_length = vid.shape[0]
        if total_length > subseq:
            subseq_length = subseq
            # start_index = torch.randint(0, total_length - subseq_length + 1, (1,))
            random_subsequence = vid[idx:idx + subseq_length]
        else:
            random_subsequence = vid
        
        return random_subsequence


def temp():
    pthI = r'/data/zhuangshuxin/datasets/2D_cineMRI_anzhen/annotations'
    pthII = r'/data/zhuangshuxin/datasets/2D_cineMRI_anzhen/videos'
    dicts = {}
    num = 0
    for case in tqdm(os.listdir(pthI)):   
        tempI = []
        for slc in os.listdir(os.path.join(pthI, case)):
            gt = nibabel.load(os.path.join(pthI, case, slc)).get_fdata()
            gt = torch.Tensor(gt)
            t,h,w = gt.shape
            mark = True
            for i in range(t):
                a = torch.unique(gt[i])
                if len(list(torch.unique(gt[i])))!=3:
                    num += 1
                    os.remove(os.path.join(pthI, case, slc))
                    break
    #         tempI.append(pthI)
    #     dicts.update({case:[element for element in tempI[0] if element in tempI[1]]})

    # save_json(dicts, os.path.join(r'/data/zhuangshuxin/datasets/M&M/val.json'))
    # pass
    print(num)



if __name__ == "__main__":

    # dataset = VOSDataset(name='shengyi', mode='train', dim='2D', view='A2C', augmentation=True)
    # vos_dataset = torch.utils.data.DataLoader(
    # dataset,
    # batch_size=1,
    # num_workers=0 ,
    # drop_last=True,
    # pin_memory=True,
    # )
    # for data in tqdm(dataset):
    #     images = data['images']
    #     lbt = data['gts']
    #     # subplots(images)
    #     # la_1st, myo_1st, lv_1st = data['split_gts'][0]


    #     # b,t,h,w = lbt.shape
    #     # subplots(lbt.reshape(b*t,h,w))
    #     # pass

    temp()
        



