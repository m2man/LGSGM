'''
This script is to extract visual feature from bounding box of objects in images
EfficientNet b5 is used by default
'''
import random
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from functools import partial
import joblib
import json
from PIL import Image
import torch.nn as nn
import os
device = torch.device('cuda:0')
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

IMAGE_DIR = 'path/to/flickr30k_images' # list of images
DATA_DIR = './Data'

list_image_size = [224, 240, 260, 300, 380, 456, 528, 600] # wrt efficientnet b0 to b7
image_size = list_image_size[5]
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
# transformList.append(transforms.RandomResizedCrop(CROP_SIZE))
transformList.append(transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC)) # Somehow this one is better
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
transform_val = transforms.Compose(transformList)

list_obj_process = [] 
list_pred_process = []

##### DATA UTILS #####
class ImageDataset(Dataset):
    def __init__(self, image_sgg):
        # Do something
        self.image_sgg = image_sgg
        self.list_image_id = list(self.image_sgg.keys())
            
    def __len__(self):
        return len(self.list_image_id)
    
    def __getitem__(self, idx):
        image_id = self.list_image_id[idx]
        bboxes = self.image_sgg[image_id]['bbox']#['bbox']
        n_obj = len(bboxes)
        return image_id, bboxes, n_obj
    
def image_collate_fn(batch, transform):
    image_id, image_bboxes, n_obj = zip(*batch)
    total_obj = sum(n_obj)
    image_obj_ft = torch.zeros(total_obj, 3, image_size, image_size)
    offset = 0
    for idx, imgid in enumerate(image_id):
        bboxes = image_bboxes[idx]
        im = Image.open(f"{IMAGE_DIR}/{imgid}").convert('RGB')
        for bbox in bboxes:  
            obj_img = im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            image_obj_ft[offset] = transform(obj_img)
            offset+=1
    return image_id, image_obj_ft,  n_obj

def make_ImageDataLoader(dataset, transform, batch_size=4, num_workers=8, pin_memory=True, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(image_collate_fn, transform=transform), pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader

# PRED
class ImagePredDataset(Dataset):
    def __init__(self, image_sgg):
        # Do something
        self.image_sgg = image_sgg
        self.list_image_id = list(self.image_sgg.keys())
            
    def __len__(self):
        return len(self.list_image_id)
    
    def __getitem__(self, idx):
        image_id = self.list_image_id[idx]
        obj_bboxes = self.image_sgg[image_id]['bbox']#['bbox']
        rels = self.image_sgg[image_id]['rels']#['sgg'] #['rels']
        n_pred = len(rels)
        bboxes = []
        for idx, rel in enumerate(rels):
            s, p, o = rel #, score = rel
            try:
                s_idx = int(s.split(':')[-1])
                o_idx = int(o.split(':')[-1])
                s_bbox = obj_bboxes[s_idx]
                o_bbox = obj_bboxes[o_idx]
            except Exception as e:
                print(e)
                print(image_id)
                print(rel)
                print(len(obj_bboxes))
            min_left = min(s_bbox[0], o_bbox[0])
            min_upper = min(s_bbox[1], o_bbox[1])
            max_right = max(s_bbox[2], o_bbox[2])
            max_lower = max(s_bbox[3], o_bbox[3])
            f_bbox = [min_left, min_upper, max_right, max_lower]
            bboxes.append(f_bbox)
        return image_id, bboxes, n_pred
    
def image_pred_collate_fn(batch, transform):
    image_id, image_bboxes, n_pred = zip(*batch)
    total_pred = sum(n_pred)
    image_pred_ft = torch.zeros(total_pred, 3, image_size, image_size)
    offset = 0
    for idx, imgid in enumerate(image_id):
        bboxes = image_bboxes[idx]
        im = Image.open(f"{IMAGE_DIR}/{imgid}").convert('RGB')
        for bbox in bboxes:  
            pred_img = im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            image_pred_ft[offset] = transform(pred_img)
            offset+=1
    return image_id, image_pred_ft,  n_pred

def make_ImagePredDataLoader(dataset, transform, batch_size=4, num_workers=8, pin_memory=True, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(image_pred_collate_fn, transform=transform), pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader

##### MODELS #####
class Visual_Feature(nn.Module):
    # Extract feature from an input images by using efficientnet
    def __init__(self, structure='b0'):
        # structure only b0 or b4
        super(Visual_Feature, self).__init__()
        self.structure = structure
        full_structure = 'efficientnet-' + self.structure
        self.backbone = EfficientNet.from_pretrained(full_structure)    
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self, inputs):
        #bs = inputs.size(0)
        x = self.backbone.extract_features(inputs)
        x = self.avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x
    
def main_obj(subset='train', effnet='b0'):
    images_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data.joblib")
    SAVE_DIR = f'{DATA_DIR}/VisualObjectFeatures_{effnet}'
    dts = ImageDataset(images_data)
    dtld = make_ImageDataLoader(dts, transform_val, batch_size=4, shuffle=False)
    visual_ft_model = Visual_Feature(effnet)
    visual_ft_model = visual_ft_model.to(device)
    visual_ft_model.eval()
    with torch.no_grad():
        for batchID, batch in enumerate(dtld):
            if batchID % 1000 == 0:
                print(f"Processing {batchID}/{len(dtld)} ...")
            list_images_id, image_obj_img, list_n_obj = batch
            if f"{list_images_id[0][:-4]}.joblib" in list_obj_process:
                continue
            image_ft = []
            n_obj = len(image_obj_img)
            for idx_obj in range(0, n_obj, 10):
                image_obj_img_b = image_obj_img[idx_obj:min(idx_obj+10, n_obj)]
                image_obj_img_b = image_obj_img_b.to(device)
                image_ft_b = visual_ft_model(image_obj_img_b)
                image_ft.append(image_ft_b)
            image_ft = torch.cat(image_ft)
            image_ft = image_ft.data.cpu().numpy()
            offset = 0
            for idx, imageid in enumerate(list_images_id):
                n_obj = list_n_obj[idx]
                temp = image_ft[offset : offset + n_obj, :]
                joblib.dump(temp, f"{SAVE_DIR}/{imageid[:-4]}.joblib")
                offset += n_obj

def main_pred(subset='train', effnet='b0'):
    images_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data.joblib")
    SAVE_DIR = f"{DATA_DIR}/VisualPredFeatures_{effnet}"
    dts = ImagePredDataset(images_data)
    dtld = make_ImagePredDataLoader(dts, transform_val, batch_size=4, shuffle=False)
    visual_ft_model = Visual_Feature(effnet)
    visual_ft_model = visual_ft_model.to(device)
    visual_ft_model.eval()
    with torch.no_grad():
        for batchID, batch in enumerate(dtld):
            if batchID % 1000 == 0:
                print(f"Processing {batchID}/{len(dtld)} ...")
            list_images_id, image_pred_img, list_n_pred = batch
            image_ft = []
            n_pred = len(image_pred_img)
            for idx_pred in range(0, n_pred, 10):
                image_pred_img_b = image_pred_img[idx_pred:min(idx_pred+10, n_pred)]
                image_pred_img_b = image_pred_img_b.to(device)
                image_ft_b = visual_ft_model(image_pred_img_b)
                image_ft.append(image_ft_b)
            image_ft = torch.cat(image_ft)
            image_ft = image_ft.data.cpu().numpy()
            
            #image_pred_img = image_pred_img.to(device)
            #image_ft = visual_ft_model(image_pred_img)
            #image_ft = image_ft.data.cpu().numpy()
            offset = 0
            for idx, imageid in enumerate(list_images_id):
                n_pred = list_n_pred[idx]
                temp = image_ft[offset : offset + n_pred, :]
                joblib.dump(temp, f"{SAVE_DIR}/{imageid[:-4]}.joblib")
                offset += n_pred
                    
print('Processing Obj ...')
main_obj(subset='train', effnet='b5')
#main_obj(subset='val', effnet='b5')
#main_obj(subset='test', effnet='b5')

print('Processing Pred ...')
main_pred(subset='train', effnet='b5')
#main_pred(subset='val', effnet='b5')
#main_pred(subset='test', effnet='b5')

