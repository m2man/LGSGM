# INCLUDE VISUAL FEATURES FOR PREDICATE (optional)
import random
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
import json
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
import joblib
import json
from functools import partial
from PIL import Image

IMAGE_DIR = 'path/to/flickr30k_images'
OBJ_FT_DIR = './VisualObjectFeatures' # run extract_visual_features.py to get this
PRED_FT_DIR = './VisualPredFeatures' # run extract_visual_features.py to get this

def indexing_sent(sent, word2idx, add_start_end=True):
    words = word_tokenize(sent)
    if add_start_end:
        words = ['<start>'] + words + ['<end>']
    words_idx = []
    for word in words:
        try:
            idx = word2idx[word]
        except:
            idx = word2idx['<unk>']
        words_idx.append(idx)
    return words_idx

def indexing_rels(rels, word2idx, add_start_end=True):
    rels_idx = []
    for rel in rels:
        for idx, word in enumerate(rel):
            if ':' in word: # rels in sentence has ":"
                word = word.split(':')[0]
                rel[idx] = word
        rel = ' '.join(rel)
        rel = word_tokenize(rel)
        if add_start_end:
            rel = ['<start>'] + rel + ['<end>']
        rel_idx = []
        for word in rel:
            try:
                idx = word2idx[word]
            except:
                idx = word2idx['<unk>']
            rel_idx.append(idx)
        rels_idx.append(rel_idx)
    return rels_idx

def encode_image_sgg_to_matrix(sgg, word2idx_obj, word2idx_pred):
    '''
    sgg is dict with rels, bbox, and labels 
    word2idx dictionary to encode word into numeric
    Return obj, pred, and edge matrix in which
    obj = [n_obj, 1] indicating index of object in obj_to_idx --> will pass to embedding
    pred = [n_pred, 1] indicating index of predicate in pred_to_idx --> will pass to embedding
    edge = [n_pred, 2] indicating the relations between objects where edge[k] = [i,j] = [obj[i], pred[k], obj[j]] relations
    '''
        
    # obj_np = np.zeros(len(isgg['bbox']['labels']), dtype=int)
    # pred_np = np.zeros(len(isgg['sgg']), dtype=int)
    # edge_np = np.zeros((len(isgg['sgg']), 2), dtype=int)
    obj_np = []
    pred_np = []
    edge_np = []
    
    try:
        sgg_rels = sgg['rels']
    except:
        sgg_rels = sgg['sgg']
    try:
        sgg_labels = sgg['labels']
    except:
        sgg_labels = sgg['bbox']['labels']
        
    for idx, obj in enumerate(sgg_labels):
        label_to_idx = word2idx_obj[obj]
        obj_np.append(label_to_idx)
    
    for idx, rel in enumerate(sgg_rels):
        sub_pos = rel[0].split(':')[1]
        pred_label = rel[1]
        # print(pred_label)
        obj_pos = rel[2].split(':')[1]
        label_to_idx = word2idx_pred[pred_label]
        pred_np.append(label_to_idx)
        edge_np.append([int(sub_pos), int(obj_pos)])
    
    obj_np = np.asarray(obj_np, dtype=int)
    pred_np = np.asarray(pred_np, dtype=int)
    edge_np = np.asarray(edge_np, dtype=int)
    
    return obj_np, pred_np, edge_np

def encode_caption_sgg_to_matrix(sgg, word2idx):
    '''
    sgg is dictionary with sent and rels
    sent and rels are lemmatised already
    Return obj, pred, and edge matrix in which
    obj = [n_obj, ] indicating index of object in obj_to_idx --> will pass to embedding
    pred = [n_pred, ] indicating index of predicate in pred_to_idx --> will pass to embedding
    edge = [n_pred, 2] indicating the relations between objects where edge[k] = [i,j] = [obj[i], pred[k], obj[j]] relations
    sent_to_idx: encoded sentence with <start> and <end> token
    '''
    
    obj_np = []
    pred_np = []
    edge_np = []
    
    # obj_np = np.zeros(len(csgg['labels']), dtype=int)
    # pred_np = np.zeros(len(csgg['sgg']), dtype=int)
    # edge_np = np.zeros((len(csgg['sgg']), 2), dtype=int)
    
    sent_to_idx = indexing_sent(sent=sgg['sent'], word2idx=word2idx, add_start_end=True) # list
    
    labels = [x[0] for x in sgg['rels']] + [x[2] for x in sgg['rels']]
    labels = np.unique(np.asarray(labels)).tolist()

    for idx, obj in enumerate(labels):
        try:
            label_to_idx = word2idx[obj]
        except:
            label_to_idx = word2idx['<unk>']
        obj_np.append(label_to_idx)   
        
    for idx, rel in enumerate(sgg['rels']):
        sub, pred_label, obj = rel[0], rel[1], rel[2]
        sub_pos = labels.index(sub)
        obj_pos = labels.index(obj)
        edge_np.append([int(sub_pos), int(obj_pos)])
    
    pred_np = indexing_rels(rels=sgg['rels'], word2idx=word2idx, add_start_end=True) # list of list
    # pred: [<start> , sub , pred, obj, <end>]
    len_pred = [len(x) for x in pred_np] # len of a pred <start> sub, pred (can be multiple words), obj <end>
    obj_np = np.asarray(obj_np, dtype=int)
    #pred_np = np.asarray(pred_np, dtype=int)
    edge_np = np.asarray(edge_np, dtype=int)
    
    return obj_np, pred_np, edge_np, len_pred, sent_to_idx # obj and edge is numpy array, other is list

# ====== IMAGE DATASET ======
# Only use for validating entire dataset
# Generate image sgg dataset only
class ImageDataset(Dataset):
    def __init__(self, image_sgg, word2idx_obj, word2idx_pred, numb_sample=None):
        # Do something
        self.image_sgg = image_sgg
        self.list_image_id = list(self.image_sgg.keys())
        self.numb_sample = numb_sample
        self.word2idx_obj = word2idx_obj
        self.word2idx_pred = word2idx_pred
        if self.numb_sample is None or self.numb_sample <= 0 or self.numb_sample > len(self.image_sgg):
            self.numb_sample = len(self.image_sgg)
            assert self.numb_sample == len(self.list_image_id)
            
    def __len__(self):
        return self.numb_sample
    
    def __getitem__(self, idx):
        image_id = self.list_image_id[idx]
        img_obj_np, img_pred_np, img_edge_np = encode_image_sgg_to_matrix(sgg=self.image_sgg[image_id],
                                                                          word2idx_obj=self.word2idx_obj,
                                                                          word2idx_pred=self.word2idx_pred)
        result = dict()
        result['id'] = image_id
        result['object'] = img_obj_np
        result['predicate'] = img_pred_np
        result['edge'] = img_edge_np
        result['numb_obj'] = len(img_obj_np)
        result['numb_pred'] = len(img_pred_np)
        result['obj_bboxes'] = self.image_sgg[image_id]['bbox']
        
        rels = self.image_sgg[image_id]['rels']
        pred_bboxes = []
        for idx, rel in enumerate(rels):
            s, p, o = rel
            s_idx = int(s.split(':')[-1])
            o_idx = int(o.split(':')[-1])
            s_bbox = result['obj_bboxes'][s_idx]
            o_bbox = result['obj_bboxes'][o_idx]
            min_left = min(s_bbox[0], o_bbox[0])
            min_upper = min(s_bbox[1], o_bbox[1])
            max_right = max(s_bbox[2], o_bbox[2])
            max_lower = max(s_bbox[3], o_bbox[3])
            f_bbox = [min_left, min_upper, max_right, max_lower]
            pred_bboxes.append(f_bbox)
        result['pred_bboxes'] = pred_bboxes
        
        return result
    
def image_collate_fn(batch, transform):
    image_obj = np.array([]) 
    image_pred = np.array([]) 
    image_edge = np.array([]) 
    image_numb_obj = np.array([]) 
    image_numb_pred = np.array([]) 
    image_obj_offset = 0
    image_obj_bboxes = []
    image_pred_bboxes = []
    image_id = []
    for ba in batch:
        image_obj = np.append(image_obj, ba['object'])
        image_pred = np.append(image_pred, ba['predicate'])
        for idx_row in range(ba['edge'].shape[0]):
            edge = ba['edge'][idx_row] + image_obj_offset
            image_edge = np.append(image_edge, edge)
        image_obj_offset += ba['numb_obj']
        image_numb_obj = np.append(image_numb_obj, ba['numb_obj'])
        image_numb_pred = np.append(image_numb_pred, ba['numb_pred'])
        image_obj_bboxes.append(ba['obj_bboxes'])
        image_pred_bboxes.append(ba['pred_bboxes'])
        image_id += [ba['id']]
    image_edge = image_edge.reshape(-1, 2)

    image_obj = torch.LongTensor(image_obj)
    image_pred = torch.LongTensor(image_pred)
    image_edge = torch.LongTensor(image_edge)
    image_numb_obj = torch.LongTensor(image_numb_obj)
    image_numb_pred = torch.LongTensor(image_numb_pred)
    
    image_obj_ft = torch.zeros(len(image_obj), 3, 224, 224)
    image_pred_ft = torch.zeros(len(image_pred), 3, 224, 224)
    offset = 0
    p_offset = 0
    for idx, imgid in enumerate(image_id):
        bboxes = image_obj_bboxes[idx]
        im = Image.open(f"{IMAGE_DIR}/{imgid}").convert('RGB')
        for bbox in bboxes:  
            obj_img = im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            image_obj_ft[offset] = transform(obj_img)
            offset+=1
        p_bboxes = image_pred_bboxes[idx]
        for p_bbox in p_bboxes:  
            pred_img = im.crop((p_bbox[0], p_bbox[1], p_bbox[2], p_bbox[3]))
            image_pred_ft[p_offset] = transform(pred_img)
            p_offset+=1
            
    return image_obj, image_obj_ft, image_pred, image_pred_ft, image_edge, image_numb_obj, image_numb_pred

def make_ImageDataLoader(dataset, transform, batch_size=4, num_workers=8, pin_memory=True, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(image_collate_fn, transform=transform), pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader



# ====== CAPTION DATASET ======
# Only use for validating entire dataset
# Generate caption sgg dataset only (sentence + sgg)
class CaptionDataset(Dataset):
    def __init__(self, caption_sgg, word2idx, numb_sample=None):
        # Do something
        self.caption_sgg = caption_sgg
        self.list_caption_id = list(self.caption_sgg.keys())
        self.numb_sample = numb_sample
        self.word2idx = word2idx
        if self.numb_sample is None or self.numb_sample <= 0 or self.numb_sample > len(self.caption_sgg):
            self.numb_sample = len(self.caption_sgg)
            assert self.numb_sample == len(self.list_caption_id)
            
    def __len__(self):
        return self.numb_sample
    
    def __getitem__(self, idx):
        caption_id = self.list_caption_id[idx]
        cap_obj_np, cap_pred_np, cap_edge_np, cap_len_pred, cap_sent_np = encode_caption_sgg_to_matrix(
            sgg=self.caption_sgg[caption_id], word2idx=self.word2idx)
        
        result = dict()
        result['object'] = cap_obj_np
        result['predicate'] = cap_pred_np
        result['edge'] = cap_edge_np
        result['sent'] = cap_sent_np
        result['numb_obj'] = len(cap_obj_np)
        result['numb_pred'] = len(cap_pred_np)
        result['len_pred'] = cap_len_pred
        
        return result

def caption_collate_fn(batch):
    caption_obj = np.array([]) 
    caption_pred = []
    caption_edge = np.array([]) 
    caption_numb_obj = []
    caption_numb_pred = []
    caption_sent = []
    caption_len_sent = []
    caption_len_pred = []
    caption_obj_offset = 0
    
    for ba in batch:
        caption_obj = np.append(caption_obj, ba['object'])
        for idx_row in range(ba['edge'].shape[0]):
            edge = ba['edge'][idx_row] + caption_obj_offset
            caption_edge = np.append(caption_edge, edge)
            caption_pred += [torch.LongTensor(ba['predicate'][idx_row])]
            
        caption_obj_offset += ba['numb_obj']
        caption_numb_obj += [ba['numb_obj']]
        caption_numb_pred += [ba['numb_pred']]
        # caption_pos_sent = np.append(caption_pos_sent, ba['caption_pos']['sent'])
        caption_sent += [torch.LongTensor(ba['sent'])]
        caption_len_sent += [len(ba['sent'])]
        caption_len_pred += ba['len_pred']
    caption_edge = caption_edge.reshape(-1, 2)
    
    caption_obj = torch.LongTensor(caption_obj)
    #caption_pred = torch.LongTensor(caption_pred)
    caption_edge = torch.LongTensor(caption_edge)
    caption_numb_obj = torch.LongTensor(caption_numb_obj)
    caption_numb_pred = torch.LongTensor(caption_numb_pred)
    
    return caption_obj, caption_pred, caption_edge,  caption_sent,\
           caption_numb_obj, caption_numb_pred, caption_len_pred, caption_len_sent

def make_CaptionDataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=caption_collate_fn, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader

# ==== PAIR IMAGES - CAPTIONS ====
# just sample a pair of matching images and captions
# Can be use for hardest negative triplet loss

# ====== IMAGE-CAPTION TRIPLET DATASET ======
class PairGraphDataset(Dataset):
    '''
    Generate pair of graphs which from image and caption
    '''
    def __init__(self, image_sgg, caption_sgg, image_caption_matching, caption_image_matching, word2idx_cap, word2idx_img_obj, word2idx_img_pred, numb_sample=None):
        '''
        image_sgg: dictionary of scene graph from images with format image_sgg[image_id]['rels'] and image_sgg[image_id]['labels']
        caption_sgg: dictionary of scene graph from captions with format caption_sgg[cap_id]['rels'] and caption_sgg[cap_id]['sent']
        Note that caption_sgg and image_sgg are all lemmatised
        image_caption_matching: dictionary describes which image matches which caption with format image_caption_matching[image_id] = [cap_id_1, cap_id_2, ...]
        caption_image_matching: reverse dictionary of above caption_image_matching[cap_id] = image_id
        word2idx: dictionary to map words into index for learning embedding
        numb_sample: int indicating number of sample in the dataset
        '''
        # Do something
        self.image_sgg = image_sgg
        self.caption_sgg = caption_sgg
        self.image_caption_matching = image_caption_matching
        self.caption_image_matching = caption_image_matching
        self.list_image_id = list(self.image_caption_matching.keys())
        self.list_caption_id = list(self.caption_image_matching.keys())
        self.numb_sample = numb_sample
        self.word2idx_cap = word2idx_cap
        self.word2idx_img_obj = word2idx_img_obj
        self.word2idx_img_pred = word2idx_img_pred
        self.list_match_pairs = []
        for caption_id in self.list_caption_id:
            image_id = self.caption_image_matching[caption_id]
            self.list_match_pairs.append((image_id, caption_id))
        # self.set_match_pairs = set(self.list_match_pairs)
        self.numb_pairs = len(self.list_match_pairs)
        
        if self.numb_sample is None:
            self.numb_sample = self.numb_pairs
            
    def create_pairs(self, seed=1509): # Have to run this function at the beginning of every epoch
        # Shuffle Item
        random.seed(seed)
        print('Creating Pairs of Graphs ...')
        sample_match = self.list_match_pairs.copy()
        if self.numb_sample <= self.numb_pairs: 
            random.shuffle(sample_match)
            sample_match = sample_match[0:self.numb_sample]
        else:
            numb_gen = self.numb_sample - self.numb_pairs
            pairs_gen = random.choices(self.list_match_pairs, k=numb_gen)
            sample_match = sample_match + pairs_gen
            random.shuffle(sample_match)
        self.samples = sample_match
    
    def __getitem__(self, i):
        # Get item
        sample = self.samples[i]
        imgid, capid = sample

        try:
            img_obj_np, img_pred_np, img_edge_np = encode_image_sgg_to_matrix(sgg=self.image_sgg[imgid],
                                                                              word2idx_obj=self.word2idx_img_obj,
                                                                              word2idx_pred=self.word2idx_img_pred)
            cap_obj_np, cap_pred_np, cap_edge_np, cap_len_pred, cap_sent_np = encode_caption_sgg_to_matrix(
                sgg=self.caption_sgg[capid], word2idx=self.word2idx_cap)
        except Exception as e:
            print(e)
            print(f"Error in {sample}")
            
        result = dict()
        result['image'] = dict()
        result['caption'] = dict()
        
        # All is numpy array
        result['image']['object'] = img_obj_np
        result['image']['predicate'] = img_pred_np
        result['image']['edge'] = img_edge_np
        result['image']['numb_obj'] = len(img_obj_np)
        result['image']['numb_pred'] = len(img_pred_np)
        result['image']['id'] = imgid
        result['image']['obj_bboxes'] = self.image_sgg[imgid]['bbox']
        rels = self.image_sgg[image_id]['rels']
        pred_bboxes = []
        for idx, rel in enumerate(rels):
            s, p, o = rel
            s_idx = int(s.split(':')[-1])
            o_idx = int(o.split(':')[-1])
            s_bbox = result['image']['obj_bboxes'][s_idx]
            o_bbox = result['image']['obj_bboxes'][o_idx]
            min_left = min(s_bbox[0], o_bbox[0])
            min_upper = min(s_bbox[1], o_bbox[1])
            max_right = max(s_bbox[2], o_bbox[2])
            max_lower = max(s_bbox[3], o_bbox[3])
            f_bbox = [min_left, min_upper, max_right, max_lower]
            pred_bboxes.append(f_bbox)
        result['image']['pred_bboxes'] = pred_bboxes
        
        
        # All is list
        result['caption']['object'] = cap_obj_np # [numpy array (numb obj)]
        result['caption']['predicate'] = cap_pred_np # [list of list]
        result['caption']['edge'] = cap_edge_np # [numpy array (numb_pred, 2)]
        result['caption']['sent'] = cap_sent_np # [list]
        result['caption']['numb_obj'] = len(cap_obj_np) # [scalar]
        result['caption']['len_pred'] = cap_len_pred # len of each predicate in a caption [list]
        result['caption']['numb_pred'] = len(cap_pred_np) # number of predicate in a caption [scalar]
        result['caption']['id'] = capid
        # result['caption']['sgg'] = self.caption_sgg[sample[1]] # for debug
        # result['image']['sgg'] = self.image_sgg[sample[0]] # for debug
        
        return result
    
    def __len__(self):
        return(len(self.samples))
    
# Collate function for preprocessing batch in dataloader
def pair_collate_fn(batch, transform):
    '''
    image obj, pred, edge is tensor
    others is list
    '''
    image_obj = np.array([]) 
    image_pred = np.array([]) 
    image_edge = np.array([]) 
    image_numb_obj = []
    image_numb_pred = []
    image_obj_offset = 0
    image_obj_bboxes = []
    image_pred_bboxes = []
        
    caption_obj = np.array([]) 
    caption_pred = []
    caption_edge = np.array([]) 
    caption_numb_obj = [] 
    caption_numb_pred = []
    caption_len_pred = []
    caption_sent = []
    caption_len_sent = []
    caption_obj_offset = 0

    caption_id = [] # for debug
    image_id = [] # for debug
    
    for ba in batch:
        # Image SGG
        image_obj = np.append(image_obj, ba['image']['object'])
        image_pred = np.append(image_pred, ba['image']['predicate'])
        for idx_row in range(ba['image']['edge'].shape[0]):
            edge = ba['image']['edge'][idx_row] + image_obj_offset
            image_edge = np.append(image_edge, edge)
        image_obj_offset += ba['image']['numb_obj']
        image_numb_obj += [ba['image']['numb_obj']]
        image_numb_pred += [ba['image']['numb_pred']]
        image_obj_bboxes.append(ba['image']['obj_bboxes'])
        image_pred_bboxes.append(ba['image']['pred_bboxes'])
        
        # Caption SGG
        caption_obj = np.append(caption_obj, ba['caption']['object'])
        for idx_row in range(ba['caption']['edge'].shape[0]):
            edge = ba['caption']['edge'][idx_row] + caption_obj_offset
            caption_pred += [torch.LongTensor(ba['caption']['predicate'][idx_row])]
            caption_edge = np.append(caption_edge, edge)
        caption_obj_offset += ba['caption']['numb_obj']
        caption_numb_obj += [ba['caption']['numb_obj']]
        caption_numb_pred += [ba['caption']['numb_pred']]
        caption_sent += [torch.LongTensor(ba['caption']['sent'])]
        caption_len_sent += [len(ba['caption']['sent'])]
        # [len p1, len p2, .. len pj (from 1st sample, j+1 pred), len pt, ...len pt+k, ...(2nd sample, k+1 pred)]
        caption_len_pred += ba['caption']['len_pred'] 
        
        image_id += [ba['image']['id']]
        caption_id += [ba['caption']['id']]
    
    # reshape edge to [n_pred, 2] size
    image_edge = image_edge.reshape(-1, 2)
    caption_edge = caption_edge.reshape(-1, 2)
    
    image_obj = torch.LongTensor(image_obj)
    image_pred = torch.LongTensor(image_pred)
    image_edge = torch.LongTensor(image_edge)
    #image_numb_obj = torch.LongTensor(image_numb_obj)
    #image_numb_pred = torch.LongTensor(image_numb_pred)
    
    caption_obj = torch.LongTensor(caption_obj)
    # caption_pred = torch.LongTensor(caption_pred)
    caption_edge = torch.LongTensor(caption_edge)
    #caption_numb_obj = torch.LongTensor(caption_numb_obj)
    #caption_numb_pred = torch.LongTensor(caption_numb_pred)
    # caption_sent = torch.LongTensor(caption_pos_sent)
    
    image_obj_ft = torch.zeros(len(image_obj), 3, 224, 224)
    image_pred_ft = torch.zeros(len(image_pred), 3, 224, 224)
    offset = 0
    p_offset = 0
    for idx, imgid in enumerate(image_id):
        bboxes = image_obj_bboxes[idx]
        im = Image.open(f"{IMAGE_DIR}/{imgid}").convert('RGB')
        for bbox in bboxes:  
            obj_img = im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            image_obj_ft[offset] = transform(obj_img)
            offset+=1
        p_bboxes = image_pred_bboxes[idx]    
        for p_bbox in bboxes:  
            pred_img = im.crop((p_bboxes[0], p_bboxes[1], p_bboxes[2], p_bboxes[3]))
            image_pred_ft[p_offset] = transform(predpred_img)
            p_offset+=1
            
    assert image_edge.shape[0] == image_pred.shape[0]
    assert caption_edge.shape[0] == sum(caption_numb_pred)

    return image_obj, image_obj_ft, image_pred, image_pred_ft, image_edge, image_numb_obj, image_numb_pred,\
           caption_obj, caption_pred, caption_edge,  caption_sent,\
           caption_numb_obj, caption_numb_pred, caption_len_pred, caption_len_sent#, image_id, caption_id

def make_PairGraphDataLoader(dataset, transform, batch_size=4, num_workers=8, pin_memory=True, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(pair_collate_fn, transform=transform), pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader


# ====== IMAGE-CAPTION TRIPLET DATASET ======
class PairGraphPrecomputeDataset(Dataset):
    '''
    Generate pair of graphs which from image and caption
    '''
    def __init__(self, image_sgg, caption_sgg, image_caption_matching, caption_image_matching, word2idx_cap, word2idx_img_obj, word2idx_img_pred, effnet='b0', numb_sample=None, obj_ft_dir=OBJ_FT_DIR, pred_ft_dir=PRED_FT_DIR):
        '''
        image_sgg: dictionary of scene graph from images with format image_sgg[image_id]['rels'] and image_sgg[image_id]['labels']
        caption_sgg: dictionary of scene graph from captions with format caption_sgg[cap_id]['rels'] and caption_sgg[cap_id]['sent']
        Note that caption_sgg and image_sgg are all lemmatised
        image_caption_matching: dictionary describes which image matches which caption with format image_caption_matching[image_id] = [cap_id_1, cap_id_2, ...]
        caption_image_matching: reverse dictionary of above caption_image_matching[cap_id] = image_id
        word2idx: dictionary to map words into index for learning embedding
        numb_sample: int indicating number of sample in the dataset
        '''
        # Do something
        self.OBJ_FT_DIR = obj_ft_dir
        self.PRED_FT_DIR = pred_ft_dir
        self.effnet = effnet
        self.image_sgg = image_sgg
        self.caption_sgg = caption_sgg
        self.image_caption_matching = image_caption_matching
        self.caption_image_matching = caption_image_matching
        self.list_image_id = list(self.image_caption_matching.keys())
        self.list_caption_id = list(self.caption_image_matching.keys())
        self.numb_sample = numb_sample
        self.word2idx_cap = word2idx_cap
        self.word2idx_img_obj = word2idx_img_obj
        self.word2idx_img_pred = word2idx_img_pred
        self.list_match_pairs = []
        for caption_id in self.list_caption_id:
            image_id = self.caption_image_matching[caption_id]
            self.list_match_pairs.append((image_id, caption_id))
        # self.set_match_pairs = set(self.list_match_pairs)
        self.numb_pairs = len(self.list_match_pairs)
        
        if self.numb_sample is None:
            self.numb_sample = self.numb_pairs
            
    def create_pairs(self, seed=1509): # Have to run this function at the beginning of every epoch
        # Shuffle Item
        random.seed(seed)
        print('Creating Pairs of Graphs ...')
        sample_match = self.list_match_pairs.copy()
        if self.numb_sample <= self.numb_pairs: 
            random.shuffle(sample_match)
            sample_match = sample_match[0:self.numb_sample]
        else:
            numb_gen = self.numb_sample - self.numb_pairs
            pairs_gen = random.choices(self.list_match_pairs, k=numb_gen)
            sample_match = sample_match + pairs_gen
            random.shuffle(sample_match)
        self.samples = sample_match
    
    def __getitem__(self, i):
        # Get item
        sample = self.samples[i]
        imgid, capid = sample

        try:
            img_obj_np, img_pred_np, img_edge_np = encode_image_sgg_to_matrix(sgg=self.image_sgg[imgid],
                                                                              word2idx_obj=self.word2idx_img_obj,
                                                                              word2idx_pred=self.word2idx_img_pred)
            cap_obj_np, cap_pred_np, cap_edge_np, cap_len_pred, cap_sent_np = encode_caption_sgg_to_matrix(
                sgg=self.caption_sgg[capid], word2idx=self.word2idx_cap)
        except Exception as e:
            print(e)
            print(f"Error in {sample}")
            
        result = dict()
        result['image'] = dict()
        result['caption'] = dict()
        
        # All is numpy array
        result['image']['object'] = img_obj_np
        result['image']['predicate'] = img_pred_np
        result['image']['edge'] = img_edge_np
        result['image']['numb_obj'] = len(img_obj_np)
        result['image']['numb_pred'] = len(img_pred_np)
        result['image']['id'] = imgid
        result['image']['object_ft'] = torch.tensor(joblib.load(f"{self.OBJ_FT_DIR}_{self.effnet}/{imgid[:-4]}.joblib")) # n_obj, ft_dim
        result['image']['pred_ft'] = torch.tensor(joblib.load(f"{self.PRED_FT_DIR}_{self.effnet}/{imgid[:-4]}.joblib")) # n_obj, ft_dim
        # All is list
        result['caption']['object'] = cap_obj_np # [numpy array (numb obj)]
        result['caption']['predicate'] = cap_pred_np # [list of list]
        result['caption']['edge'] = cap_edge_np # [numpy array (numb_pred, 2)]
        result['caption']['sent'] = cap_sent_np # [list]
        result['caption']['numb_obj'] = len(cap_obj_np) # [scalar]
        result['caption']['len_pred'] = cap_len_pred # len of each predicate in a caption [list]
        result['caption']['numb_pred'] = len(cap_pred_np) # number of predicate in a caption [scalar]
        result['caption']['id'] = capid
        # result['caption']['sgg'] = self.caption_sgg[sample[1]] # for debug
        # result['image']['sgg'] = self.image_sgg[sample[0]] # for debug
        
        return result
    
    def __len__(self):
        return(len(self.samples))
    
# Collate function for preprocessing batch in dataloader
def pair_precompute_collate_fn(batch):
    '''
    image obj, pred, edge is tensor
    others is list
    '''
    image_obj = np.array([]) 
    image_pred = np.array([]) 
    image_edge = np.array([]) 
    image_numb_obj = []
    image_numb_pred = []
    image_obj_offset = 0
    image_obj_ft = []
    image_pred_ft = []
    
    caption_obj = np.array([]) 
    caption_pred = []
    caption_edge = np.array([]) 
    caption_numb_obj = [] 
    caption_numb_pred = []
    caption_len_pred = []
    caption_sent = []
    caption_len_sent = []
    caption_obj_offset = 0

    caption_id = [] # for debug
    image_id = [] # for debug
    
    for ba in batch:
        # Image SGG
        image_obj = np.append(image_obj, ba['image']['object'])
        image_pred = np.append(image_pred, ba['image']['predicate'])
        for idx_row in range(ba['image']['edge'].shape[0]):
            edge = ba['image']['edge'][idx_row] + image_obj_offset
            image_edge = np.append(image_edge, edge)
        image_obj_offset += ba['image']['numb_obj']
        image_numb_obj += [ba['image']['numb_obj']]
        image_numb_pred += [ba['image']['numb_pred']]
        image_obj_ft.append(ba['image']['object_ft'])
        image_pred_ft.append(ba['image']['pred_ft'])
        
        # Caption SGG
        caption_obj = np.append(caption_obj, ba['caption']['object'])
        for idx_row in range(ba['caption']['edge'].shape[0]):
            edge = ba['caption']['edge'][idx_row] + caption_obj_offset
            caption_pred += [torch.LongTensor(ba['caption']['predicate'][idx_row])]
            caption_edge = np.append(caption_edge, edge)
        caption_obj_offset += ba['caption']['numb_obj']
        caption_numb_obj += [ba['caption']['numb_obj']]
        caption_numb_pred += [ba['caption']['numb_pred']]
        caption_sent += [torch.LongTensor(ba['caption']['sent'])]
        caption_len_sent += [len(ba['caption']['sent'])]
        # [len p1, len p2, .. len pj (from 1st sample, j+1 pred), len pt, ...len pt+k, ...(2nd sample, k+1 pred)]
        caption_len_pred += ba['caption']['len_pred'] 
        
        image_id += [ba['image']['id']]
        caption_id += [ba['caption']['id']]
    
    # reshape edge to [n_pred, 2] size
    image_edge = image_edge.reshape(-1, 2)
    caption_edge = caption_edge.reshape(-1, 2)
    
    image_obj = torch.LongTensor(image_obj)
    image_pred = torch.LongTensor(image_pred)
    image_edge = torch.LongTensor(image_edge)
    #image_numb_obj = torch.LongTensor(image_numb_obj)
    #image_numb_pred = torch.LongTensor(image_numb_pred)
    
    caption_obj = torch.LongTensor(caption_obj)
    # caption_pred = torch.LongTensor(caption_pred)
    caption_edge = torch.LongTensor(caption_edge)
    #caption_numb_obj = torch.LongTensor(caption_numb_obj)
    #caption_numb_pred = torch.LongTensor(caption_numb_pred)
    # caption_sent = torch.LongTensor(caption_pos_sent)
    
    image_obj_ft = torch.cat(image_obj_ft, dim=0) # tensor [total_obj, dim]
    image_pred_ft = torch.cat(image_pred_ft, dim=0) # tensor [total_pred, dim]
            
    assert image_edge.shape[0] == image_pred.shape[0]
    assert caption_edge.shape[0] == sum(caption_numb_pred)

    return image_obj, image_obj_ft, image_pred, image_pred_ft, image_edge, image_numb_obj, image_numb_pred,\
           caption_obj, caption_pred, caption_edge,  caption_sent,\
           caption_numb_obj, caption_numb_pred, caption_len_pred, caption_len_sent#, image_id, caption_id

def make_PairGraphPrecomputeDataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pair_precompute_collate_fn, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader

# ====== IMAGE DATASET ======
# Only use for validating entire dataset
# Generate image sgg dataset only
class ImagePrecomputeDataset(Dataset):
    def __init__(self, image_sgg, word2idx_obj, word2idx_pred, effnet='b0', numb_sample=None, obj_ft_dir=OBJ_FT_DIR, pred_ft_dir=PRED_FT_DIR):
        # Do something
        self.OBJ_FT_DIR = obj_ft_dir
        self.PRED_FT_DIR = pred_ft_dir
        self.effnet = effnet
        self.image_sgg = image_sgg
        self.list_image_id = list(self.image_sgg.keys())
        self.numb_sample = numb_sample
        self.word2idx_obj = word2idx_obj
        self.word2idx_pred = word2idx_pred
        if self.numb_sample is None or self.numb_sample <= 0 or self.numb_sample > len(self.image_sgg):
            self.numb_sample = len(self.image_sgg)
            assert self.numb_sample == len(self.list_image_id)
            
    def __len__(self):
        return self.numb_sample
    
    def __getitem__(self, idx):
        image_id = self.list_image_id[idx]
        img_obj_np, img_pred_np, img_edge_np = encode_image_sgg_to_matrix(sgg=self.image_sgg[image_id],
                                                                          word2idx_obj=self.word2idx_obj,
                                                                          word2idx_pred=self.word2idx_pred)
        result = dict()
        result['id'] = image_id
        result['object'] = img_obj_np
        result['predicate'] = img_pred_np
        result['edge'] = img_edge_np
        result['numb_obj'] = len(img_obj_np)
        result['numb_pred'] = len(img_pred_np)
        result['object_ft'] = torch.tensor(joblib.load(f"{self.OBJ_FT_DIR}_{self.effnet}/{image_id[:-4]}.joblib")) # n_obj, ft_dim
        result['pred_ft'] = torch.tensor(joblib.load(f"{self.PRED_FT_DIR}_{self.effnet}/{image_id[:-4]}.joblib")) # n_pred, ft_dim
        return result
    
def image_precompute_collate_fn(batch):
    image_obj = np.array([]) 
    image_pred = np.array([]) 
    image_edge = np.array([]) 
    image_numb_obj = np.array([]) 
    image_numb_pred = np.array([]) 
    image_obj_offset = 0
    image_obj_ft = []
    image_pred_ft = []
    image_id = []
    for ba in batch:
        image_obj = np.append(image_obj, ba['object'])
        image_pred = np.append(image_pred, ba['predicate'])
        for idx_row in range(ba['edge'].shape[0]):
            edge = ba['edge'][idx_row] + image_obj_offset
            image_edge = np.append(image_edge, edge)
        image_obj_offset += ba['numb_obj']
        image_numb_obj = np.append(image_numb_obj, ba['numb_obj'])
        image_numb_pred = np.append(image_numb_pred, ba['numb_pred'])
        image_obj_ft.append(ba['object_ft'])
        image_pred_ft.append(ba['pred_ft'])
        image_id += [ba['id']]
    image_edge = image_edge.reshape(-1, 2)

    image_obj = torch.LongTensor(image_obj)
    image_pred = torch.LongTensor(image_pred)
    image_edge = torch.LongTensor(image_edge)
    image_numb_obj = torch.LongTensor(image_numb_obj)
    image_numb_pred = torch.LongTensor(image_numb_pred)
    
    image_obj_ft = torch.cat(image_obj_ft, dim=0)
    image_pred_ft = torch.cat(image_pred_ft, dim=0)
            
    return image_obj, image_obj_ft, image_pred, image_pred_ft, image_edge, image_numb_obj, image_numb_pred

def make_ImagePrecomputeDataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=image_precompute_collate_fn, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader
