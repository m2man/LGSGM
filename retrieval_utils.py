import numpy as np
import torch
from metrics import xattn_score_t2i, xattn_score_i2t, CosineSimilarity
from collections import OrderedDict
from torch.autograd import Variable

def shard_xattn_t2i(images, obj_nums, captions, caplens, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images)-1)/shard_size) + 1
    n_cap_shard = int((len(captions)-1)/shard_size) + 1
    
    d = np.zeros((len(images), len(captions)))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
            for j in range(n_cap_shard):
                cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
                im = torch.from_numpy(images[im_start:im_end]).cuda()
                im_l =obj_nums[im_start:im_end]
                s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
                l = caplens[cap_start:cap_end]
                sim = xattn_score_t2i(im, im_l, s, l)
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    return d


def shard_xattn_i2t(images, obj_nums, captions, caplens, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = int((len(images)-1)/shard_size) + 1
    n_cap_shard = int((len(captions)-1)/shard_size) + 1
    
    d = np.zeros((len(images), len(captions)))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
            for j in range(n_cap_shard):
                cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
                im = torch.from_numpy(images[im_start:im_end]).cuda()
                im_l =obj_nums[im_start:im_end]
                s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
                l = caplens[cap_start:cap_end]
                sim = xattn_score_i2t(im, im_l, s, l)
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    return d

def shard_cosinesimilarity(images_geb, captions_geb, shard_size=128):
    #input is numpy format
    n_img = len(images_geb)
    n_cap = len(captions_geb)
    n_im_shard = int((n_img-1)/shard_size) + 1
    n_cap_shard = int((n_cap-1)/shard_size) + 1
    d = np.zeros((n_img, n_cap))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size*i, min(shard_size*(i+1), n_img)
            for j in range(n_cap_shard):
                cap_start, cap_end = shard_size*j, min(shard_size*(j+1), n_cap)
                img_geb = torch.from_numpy(images_geb[im_start:im_end]).cuda()
                cap_geb = torch.from_numpy(captions_geb[cap_start:cap_end]).cuda()
                sim = CosineSimilarity(img_geb, cap_geb)
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    return d

def evalrank(img_embs, obj_nums, cap_embs, cap_lens, pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, 
             cross_attn='i2t', predicate_score_rate=1, img_geb=None, cap_geb=None, disc_score=None):
    print('Contrastive Score ...')
    if cross_attn == 't2i':
        sims1 = shard_xattn_t2i(img_embs, obj_nums, cap_embs, cap_lens, shard_size=128)
        sims2 = shard_xattn_t2i(pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, shard_size=128)
        sims = sims1 + predicate_score_rate * sims2
    elif cross_attn == 'i2t':
        sims1 = shard_xattn_i2t(img_embs, obj_nums, cap_embs, cap_lens, shard_size=128)
        sims2 = shard_xattn_i2t(pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, shard_size=128)
        sims = sims1 + predicate_score_rate * sims2
    
    print('GraphEmb Score ...')
    if img_geb is not None and cap_geb is not None:
        cos_sims = shard_cosinesimilarity(img_geb, cap_geb, shard_size=256)
        sims = sims + cos_sims
    if disc_score is not None:
        sims = sims + disc_score
        
    r, rt, i2t_results = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
    ri, rti, t2i_results = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.4f" % rsum)
    print("Average i2t Recall: %.4f" % ar)
    print("Image to text: %.4f %.4f %.4f %.4f %.4f" % r)
    print("Average t2i Recall: %.4f" % ari)
    print("Text to image: %.4f %.4f %.4f %.4f %.4f" % ri)
    return rsum, r, ri

def evalrank_result(img_embs, obj_nums, cap_embs, cap_lens, pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, 
             cross_attn='i2t', predicate_score_rate=1, img_geb=None, cap_geb=None, disc_score=None):
    print('Contrastive Score ...')
    if cross_attn == 't2i':
        sims1 = shard_xattn_t2i(img_embs, obj_nums, cap_embs, cap_lens, shard_size=128)
        sims2 = shard_xattn_t2i(pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, shard_size=128)
        sims = sims1 + predicate_score_rate * sims2
    elif cross_attn == 'i2t':
        sims1 = shard_xattn_i2t(img_embs, obj_nums, cap_embs, cap_lens, shard_size=128)
        sims2 = shard_xattn_i2t(pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, shard_size=128)
        sims = sims1 + predicate_score_rate * sims2
    
    print('GraphEmb Score ...')
    if img_geb is not None and cap_geb is not None:
        cos_sims = shard_cosinesimilarity(img_geb, cap_geb, shard_size=256)
        sims = sims + cos_sims
    if disc_score is not None:
        sims = sims + disc_score
        
    r, rt, i2t_results = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
    ri, rti, t2i_results = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.4f" % rsum)
    print("Average i2t Recall: %.4f" % ar)
    print("Image to text: %.4f %.4f %.4f %.4f %.4f" % r)
    print("Average t2i Recall: %.4f" % ari)
    print("Text to image: %.4f %.4f %.4f %.4f %.4f" % ri)
    result = {}
    result['i2t'] = i2t_results
    result['t2i'] = t2i_results
    return rsum, r, ri, result
    
# find caption
def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    results = []
    for index in range(npts):
        result =dict()
        result['id'] = index
        inds = np.argsort(sims[index])[::-1]
        result['top5'] = list(inds[:5])
        result['top1'] = inds[0]
        result['top10'] = list(inds[:10])
        result['ranks'] = []
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            result['ranks'].append((i, tmp))
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

        if rank<1:
            result['is_top1'] = 1
        else:
            result['is_top1'] = 0
        if rank<5:
            result['is_top5'] = 1
        else:
            result['is_top5'] = 0

        results.append(result)

    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1), results
    else:
        return (r1, r5, r10, medr, meanr), results

# find images
def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T # ncap, nimg
    results = []
    for index in range(npts):
        for i in range(5):
            result = dict()
            result['id'] = 5*index+i
            inds = np.argsort(sims[5 * index + i])[::-1]
            result['top5'] = list(inds[:5])
            result['top10'] = list(inds[:10])
            result['top1'] = inds[0]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

            if ranks[5*index+i]<1:
                result['is_top1'] = 1
            else:
                result['is_top1'] = 0

            if ranks[5*index+i] <5:
                result['is_top5'] =1
            else:
                result['is_top5'] = 0
            result['ranks'] = [(index, ranks[5*index+i])]
            results.append(result)

    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1),results
    else:
        return (r1, r5, r10, medr, meanr), results
    
def retrieval_t2i(img_obj_emb, img_numb_o_all, cap_sent_emb, cap_len_sent_all, 
                      img_pred_emb, img_numb_p_all, cap_rels_emb, cap_numb_rels_all,
                      predicate_score_rate=1, img_geb=None, cap_geb=None, only_cosine=False):
    if img_geb is None or cap_geb is None:
        only_cosine = False
    
    if not only_cosine:
        sims1 = shard_xattn_t2i(img_obj_emb, img_numb_o_all, cap_sent_emb, cap_len_sent_all, shard_size=128)
        sims2 = shard_xattn_t2i(img_pred_emb, img_numb_p_all, cap_rels_emb, cap_numb_rels_all, shard_size=128)
        sims = sims1 + predicate_score_rate * sims2
    else:
        if img_geb is not None and cap_geb is not None:
            cos_sims = shard_cosinesimilarity(img_geb, cap_geb, shard_size=256)
            sims = sims + cos_sims

    sims = sims.T # ncap, nimg
    n_cap = sims.shape[0]
    result = []
    for index in range(n_cap):
        inds = list(np.argsort(sims[index])[::-1])
        result.append(inds)
    return result # return index of images (not the imageid)