'''
Calculate recall@k
'''
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.ranking import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
device = torch.device('cuda:0')
from torch.autograd import Variable

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def calculate_metric(gtnp, pdnp):
    # input are numpy vector
    o_pdnp = np.copy(pdnp) # this is for AUROC score
    pdnp[pdnp>=0.5] = 1
    pdnp[pdnp!=1] = 0
    total_samples = len(gtnp)
    #print(f"Total sample: {total_samples}")
    total_correct = np.sum(gtnp == pdnp)
    accuracy = total_correct / total_samples
    gt_pos = np.where(gtnp == 1)[0]
    gt_neg = np.where(gtnp == 0)[0]
    TP = np.sum(pdnp[gt_pos])
    TN = np.sum(1 - pdnp[gt_neg])
    FP = np.sum(pdnp[gt_neg])
    FN = np.sum(1 - pdnp[gt_pos])
    precision = TP / (TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    metrics = {}
    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['tp'] = int(TP)
    metrics['tn'] = int(TN)
    metrics['fp'] = int(FP)
    metrics['fn'] = int(FN)
    try:
        metrics['auc'] = roc_auc_score(gtnp, o_pdnp)
    except Exception as e:
        print(e)
        metrics['auc'] = -1
    return metrics

def find_recall(list_query, list_source, result_matching, label_matching, k=1):
    '''
    list_query, list_source: id of query and source (list)
    result_matching: index of above list predict to be matching to each other (list of list)
    label_matching: groundtruth id matching between query: source (dict)
    '''
    assert len(list_query) == len(result_matching)
    result = []
    for idx, query in enumerate(list_query):
        flag = 0
        topk = result_matching[idx][:k]
        query_idx = topk[0][0]
        source_idx = [x[1] for x in topk]
        assert query_idx == idx
        query_id = list_query[query_idx]
        source_id = [list_source[x] for x in source_idx]
        label_id = label_matching[query_id]
        for sid in source_id:
            if sid in label_id:
                flag = 1
                break
        result.append(flag)
    result = np.asarray(result)
    recall = np.sum(result) / len(result)
    return recall

# ===== FROM SGM =====
def xattn_score_t2i(images, obj_nums, captions, cap_lens):
    """
    Images: (n_image, max_n_objs, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    obj_nums:(n_obj) list of obj num per image
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    max_n_word = captions.size(1)

    ## padding caption use 0
    # for i in range(n_caption):
    #     n_word = cap_lens[i]
    #     captions[i,n_word:, :] = torch.zeros(max_n_word-n_word, captions.size(2), dtype= captions.dtype).cuda()

    cap_lens = torch.tensor(cap_lens, dtype=captions.dtype)
    cap_lens = Variable(cap_lens).to(device)
    captions = torch.transpose(captions, 1, 2)
    for i in range(n_image):
        n_obj = obj_nums[i]
        if n_obj == 0:
            img_i = images[i, :, :].unsqueeze(0).contiguous()
        else:
            img_i = images[i, : n_obj, :].unsqueeze(0).contiguous()
        # --> (n_caption , n_region ,d)
        img_i_expand = img_i.repeat(n_caption, 1, 1)
        # --> (n_caption, d, max_n_word)
        dot = torch.bmm(img_i_expand, captions)
        # if opt.clamp:
        #     dot = torch.clamp(dot, min=0)
        dot = dot.max(dim=1, keepdim=True)[0].squeeze()
        dot = dot.view(n_caption, -1).contiguous()
        dot = dot.sum(dim=1, keepdim=True)
        cap_lens = cap_lens.contiguous().view(-1,1)
        dot = dot/(cap_lens+1e-6)
        # dot = dot.mean(dim=1, keepdim=True)
        dot = torch.transpose(dot, 0, 1)
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 0)
    
    return similarities


def xattn_score_i2t(images, obj_nums, captions, cap_lens):
    """
    Images: (batch_size, max_n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    obj_nums:(n_obj) list of obj num per image
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    max_n_obj = images.size(1)

    # # padding iamge use 0
    # for i in range(n_image):
    #     n_obj = obj_nums[i]
    #     images[i, n_obj:,:]  = torch.zeros(max_n_obj-n_obj, images.size(2), dtype=images.dtype).cuda()

    obj_nums = torch.tensor(obj_nums, dtype=images.dtype)
    obj_nums = Variable(obj_nums).to(device)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        if n_word == 0:
            cap_i = captions[i, :, :].unsqueeze(0).contiguous()
        else:
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        cap_i_expand = cap_i_expand.contiguous()
        cap_i_expand = torch.transpose(cap_i_expand, 1,2)
        dot = torch.bmm(images, cap_i_expand)
        # if opt.clamp:
        #     dot = torch.clamp(dot, min=0)
        dot = dot.max(dim=2, keepdim=True)[0].squeeze()
        dot = dot.sum(dim=1, keepdim=True)
        obj_nums = obj_nums.contiguous().view(-1,1)
        dot = dot/(obj_nums+1e-6)
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def CosineSimilarity(images_geb, captions_geb):
    similarities = sim_matrix(images_geb, captions_geb) # n_img, n_caption
    return similarities

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, predicate_score_rate=1, margin=0, max_violation=False, cross_attn='i2t'):
        super(ContrastiveLoss, self).__init__()
        self.predicate_score_rate = predicate_score_rate
        self.margin = margin
        self.max_violation = max_violation
        self.cross_attn = cross_attn

    def forward(self, im, im_l, s, s_l, pred, pred_l, s_pred, s_pred_l):
        # compute image-sentence score matrix
        if self.cross_attn == 't2i':
            scores1 = xattn_score_t2i(im, im_l, s, s_l)
            scores2 = xattn_score_t2i(pred, pred_l, s_pred, s_pred_l)
            scores = scores1 + self.predicate_score_rate*scores2 

        elif self.cross_attn == 'i2t':
            scores1 = xattn_score_i2t(im, im_l, s, s_l)
            scores2 = xattn_score_i2t(pred, pred_l, s_pred, s_pred_l)
            scores = scores1 + self.predicate_score_rate*scores2    
        else:
            raise ValueError("unknown first norm type")

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.mean() + cost_im.mean()
    
class ContrastiveLoss_CosineSimilarity(nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss_CosineSimilarity, self).__init__()
        self.max_violation = max_violation
        self.margin = margin
        
    def forward(self, images_geb, captions_geb):
        scores = CosineSimilarity(images_geb, captions_geb)
        diagonal = scores.diag().view(len(images_geb), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.mean() + cost_im.mean()
    
######################################## OLD CODE ########################################
class RankingLoss(nn.Module):
    """Ranking Loss
        reduction: Reduction method to apply, return mean over batch if 'mean',
        return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, margin=1, reduction='mean'):
        super(RankingLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.cos_sim = self.cos_sim.to(device)
        self.reduction = reduction
        self.margin = margin
        
    def forward(self, image_pos, caption_pos, image_neg, caption_neg):
        '''
        all input is tensor with shape of [batch, dim]
        '''
        
        image_anchor = self.margin - (self.cos_sim(image_pos, caption_pos) - self.cos_sim(image_pos, caption_neg))
        caption_anchor = self.margin - (self.cos_sim(caption_pos, image_pos) - self.cos_sim(caption_pos, image_neg))
        
        image_anchor = torch.clamp(image_anchor, min=0)
        caption_anchor = torch.clamp(caption_anchor, min=0)
        
        loss = image_anchor + caption_anchor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
            
class MatrixLoss(nn.Module):
    """
    Cross Graph Loss
    """
    def __init__(self, margin=1, obj_coef=1, pred_coef=1, reduction='mean'):
        super(MatrixLoss, self).__init__()
        self.reduction = reduction
        self.margin = margin
        self.obj_coef = obj_coef
        self.pred_coef = pred_coef
        
    def forward(self, image_pos_obj, image_pos_pred, image_pos_numb_obj, image_pos_numb_pred, \
                caption_pos_sent, caption_pos_rels, caption_pos_len_sent, caption_pos_numb_pred, \
                image_neg_obj, image_neg_pred, image_neg_numb_obj, image_neg_numb_pred, \
                caption_neg_sent, caption_neg_rels, caption_neg_len_sent, caption_neg_numb_pred):
        '''
        image_pos_obj [total numb_obj, dim]
        image_pos_pred [total numb_pred, dim]
        image_pos_numb_obj [batch] numb obj in each image in the batch
        image_pos_numb_pred [batch] numb pred in each image in the batch
        caption_pos_sent [batch, max seq len, dim] embeding words of each caption in the batch
        caption_pos_rels [total numb_pred, dim] embedding rels in all captions
        caption_pos_len_sent [batch] len of each caption in the batch
        caption_pos_numb_pred [batch] numb pred (number of rels) in each caption in the batch
        '''
        batch_size = len(image_pos_numb_pred)

        S_nodes_ip_cp = torch.empty(batch_size).to(device)
        S_edges_ip_cp = torch.empty(batch_size).to(device)
        
        S_nodes_ip_cn = torch.empty(batch_size).to(device)
        S_edges_ip_cn = torch.empty(batch_size).to(device)
        
        S_nodes_in_cp = torch.empty(batch_size).to(device)
        S_edges_in_cp = torch.empty(batch_size).to(device)
        
        S_nodes_cp_ip = torch.empty(batch_size).to(device)
        S_edges_cp_ip = torch.empty(batch_size).to(device)
        
        S_nodes_cp_in = torch.empty(batch_size).to(device)
        S_edges_cp_in = torch.empty(batch_size).to(device)
        
        S_nodes_cn_ip = torch.empty(batch_size).to(device)
        S_edges_cn_ip = torch.empty(batch_size).to(device)
        
        ip_count_obj = 0
        ip_count_pred = 0
        cp_count_pred = 0
        in_count_obj = 0
        in_count_pred = 0
        cn_count_pred = 0
        
        for idx_batch in range(batch_size):
            ##### Image Pos - Caption Pos (IP - CP) #####
            # Node to Node comparison
            if image_pos_numb_obj[idx_batch] == 0:
                S_nodes_ip_cp[idx_batch] = 0
                S_nodes_cp_ip[idx_batch] = 0
            else:
                img_obj = image_pos_obj[ip_count_obj:(ip_count_obj+image_pos_numb_obj[idx_batch]),:] # numb_obj, dim
                sent = caption_pos_sent[idx_batch, 0:caption_pos_len_sent[idx_batch],:] # len sent, dim
                
                Node_ip_cp_m = torch.matmul(img_obj, sent.T) # numb_obj, len sent
                Node_ip_cp = torch.max(Node_ip_cp_m , dim=1)[0] # numb_obj
                Node_ip_cp = torch.sum(Node_ip_cp, dim=0) # scalar
                Node_ip_cp = Node_ip_cp / image_pos_numb_obj[idx_batch]
                S_nodes_ip_cp[idx_batch] = Node_ip_cp
                Node_cp_ip = torch.max(Node_ip_cp_m , dim=0)[0] # length sent
                Node_cp_ip = torch.sum(Node_cp_ip, dim=0) # scalar
                Node_cp_ip = Node_cp_ip / caption_pos_len_sent[idx_batch]
                S_nodes_cp_ip[idx_batch] = Node_cp_ip
                '''
                Nodes_ip_cp = torch.matmul(sent, img_obj.T) # len sent, numb_obj
                Nodes_cp_ip = Nodes_ip_cp.T #torch.matmul(img_obj, sent.T) # numb_obj, len sent
                S_nodes_ip_cp[idx_batch] = torch.mean(torch.max(Nodes_ip_cp , dim=1)[0])
                S_nodes_cp_ip[idx_batch] = torch.mean(torch.max(Nodes_cp_ip , dim=1)[0])
                '''
            # Edge to Edge comparison
            if caption_pos_numb_pred[idx_batch] == 0 or image_pos_numb_pred[idx_batch] == 0:
                S_edges_ip_cp[idx_batch] = 0
                S_edges_cp_ip[idx_batch] = 0
            else:
                img_pred = image_pos_pred[ip_count_pred:(ip_count_pred+image_pos_numb_pred[idx_batch]),:] # img_numb_pred, dim
                rels = caption_pos_rels[cp_count_pred:(cp_count_pred+caption_pos_numb_pred[idx_batch]),:] # cap_numb_pred, dim
                
                Edges_ip_cp_m = torch.matmul(img_pred, rels.T) # numb_pred, cap numb_pred
                Edges_ip_cp = torch.max(Edges_ip_cp_m , dim=1)[0] # numb_pred
                Edges_ip_cp = torch.sum(Edges_ip_cp, dim=0) # scalar
                Edges_ip_cp = Edges_ip_cp / image_pos_numb_pred[idx_batch]
                S_edges_ip_cp[idx_batch] = Edges_ip_cp
                Edges_cp_ip = torch.max(Edges_ip_cp_m , dim=0)[0] # cap numb_pred
                Edges_cp_ip = torch.sum(Edges_cp_ip, dim=0) # scalar
                Edges_cp_ip = Edges_cp_ip / caption_pos_numb_pred[idx_batch]
                S_edges_cp_ip[idx_batch] = Edges_cp_ip
                '''
                Edges_ip_cp = torch.matmul(rels, img_pred.T) # cap_numb_pred, img_numb_pred
                Edges_cp_ip = Edges_ip_cp.T #torch.matmul(img_pred, rels.T) # img_numb_pred, cap_numb_pred
                S_edges_ip_cp[idx_batch] = torch.mean(torch.max(Edges_ip_cp , dim=1)[0])
                S_edges_cp_ip[idx_batch] = torch.mean(torch.max(Edges_cp_ip , dim=1)[0])
                '''
            ##### IP - CN #####
            # Node to Node comparison
            if image_pos_numb_obj[idx_batch] == 0:
                S_nodes_ip_cn[idx_batch] = 0
                S_nodes_cn_ip[idx_batch] = 0
            else:
                img_obj = image_pos_obj[ip_count_obj:(ip_count_obj+image_pos_numb_obj[idx_batch]),:] # numb_obj, dim
                sent = caption_neg_sent[idx_batch, 0:caption_neg_len_sent[idx_batch],:] # len sent, dim
                
                Node_ip_cn_m = torch.matmul(img_obj, sent.T) # numb_obj, len sent
                Node_ip_cn = torch.max(Node_ip_cn_m , dim=1)[0] # numb_obj
                Node_ip_cn = torch.sum(Node_ip_cn, dim=0) # scalar
                Node_ip_cn = Node_ip_cn / image_pos_numb_obj[idx_batch]
                S_nodes_ip_cn[idx_batch] = Node_ip_cn
                Node_cn_ip = torch.max(Node_ip_cn_m , dim=0)[0] # length sent
                Node_cn_ip = torch.sum(Node_cn_ip, dim=0) # scalar
                Node_cn_ip = Node_cn_ip / caption_neg_len_sent[idx_batch]
                S_nodes_cn_ip[idx_batch] = Node_cn_ip
                '''
                Nodes_ip_cn = torch.matmul(sent, img_obj.T) # len sent, numb_obj
                Nodes_cn_ip = Nodes_ip_cn.T #torch.matmul(img_obj, sent.T)
                S_nodes_ip_cn[idx_batch] = torch.mean(torch.max(Nodes_ip_cn , dim=1)[0])
                S_nodes_cn_ip[idx_batch] = torch.mean(torch.max(Nodes_cn_ip , dim=1)[0])
                '''
            # Edge to Edge comparison
            if caption_neg_numb_pred[idx_batch] == 0 or image_pos_numb_pred[idx_batch] == 0:
                S_edges_ip_cn[idx_batch] = 0
                S_edges_cn_ip[idx_batch] = 0
            else:
                img_pred = image_pos_pred[ip_count_pred:(ip_count_pred+image_pos_numb_pred[idx_batch]),:] # img_numb_pred, dim
                rels = caption_neg_rels[cn_count_pred:(cn_count_pred+caption_neg_numb_pred[idx_batch]),:] # cap_numb_pred, dim
                
                Edges_ip_cn_m = torch.matmul(img_pred, rels.T) # numb_pred, cap numb_pred
                Edges_ip_cn = torch.max(Edges_ip_cn_m , dim=1)[0] # numb_pred
                Edges_ip_cn = torch.sum(Edges_ip_cn, dim=0) # scalar
                Edges_ip_cn = Edges_ip_cn / image_pos_numb_pred[idx_batch]
                S_edges_ip_cn[idx_batch] = Edges_ip_cn
                Edges_cn_ip = torch.max(Edges_ip_cn_m , dim=0)[0] # cap numb_pred
                Edges_cn_ip = torch.sum(Edges_cn_ip, dim=0) # scalar
                Edges_cn_ip = Edges_cn_ip / caption_neg_numb_pred[idx_batch]
                S_edges_cn_ip[idx_batch] = Edges_cn_ip
                '''
                Edges_ip_cn = torch.matmul(rels, img_pred.T) # cap_numb_pred, img_numb_pred
                Edges_cn_ip = Edges_ip_cn.T #torch.matmul(img_pred, rels.T)
                S_edges_ip_cn[idx_batch] = torch.mean(torch.max(Edges_ip_cn , dim=1)[0])
                S_edges_cn_ip[idx_batch] = torch.mean(torch.max(Edges_cn_ip , dim=1)[0])
                '''
            ##### IN - CP #####
            # Node to Node comparison
            if image_neg_numb_obj[idx_batch] == 0:
                S_nodes_in_cp[idx_batch] = 0
                S_nodes_cp_in[idx_batch] = 0
            else:
                img_obj = image_neg_obj[in_count_obj:(in_count_obj+image_neg_numb_obj[idx_batch]),:] # numb_obj, dim
                sent = caption_pos_sent[idx_batch, 0:caption_pos_len_sent[idx_batch],:] # len sent, dim
                
                Node_in_cp_m = torch.matmul(img_obj, sent.T) # numb_obj, len sent
                Node_in_cp = torch.max(Node_in_cp_m , dim=1)[0] # numb_obj
                Node_in_cp = torch.sum(Node_in_cp, dim=0) # scalar
                Node_in_cp = Node_in_cp / image_neg_numb_obj[idx_batch]
                S_nodes_in_cp[idx_batch] = Node_in_cp
                Node_cp_in = torch.max(Node_in_cp_m , dim=0)[0] # length sent
                Node_cp_in = torch.sum(Node_cp_in, dim=0) # scalar
                Node_cp_in = Node_cp_in / caption_pos_len_sent[idx_batch]
                S_nodes_cp_in[idx_batch] = Node_cp_in
                '''
                Nodes_in_cp = torch.matmul(sent, img_obj.T) # len sent, numb_obj
                Nodes_cp_in = Nodes_in_cp.T #torch.matmul(img_obj, sent.T)
                S_nodes_in_cp[idx_batch] = torch.mean(torch.max(Nodes_in_cp , dim=1)[0])
                S_nodes_cp_in[idx_batch] = torch.mean(torch.max(Nodes_cp_in , dim=1)[0])
                '''
            # Edge to Edge comparison
            if caption_pos_numb_pred[idx_batch] == 0 or image_neg_numb_pred[idx_batch] == 0:
                S_edges_in_cp[idx_batch] = 0
                S_edges_cp_in[idx_batch] = 0
            else:
                img_pred = image_neg_pred[in_count_pred:(in_count_pred+image_neg_numb_pred[idx_batch]),:] # img_numb_pred, dim
                rels = caption_pos_rels[cp_count_pred:(cp_count_pred+caption_pos_numb_pred[idx_batch]),:] # cap_numb_pred, dim
                
                Edges_in_cp_m = torch.matmul(img_pred, rels.T) # numb_pred, cap numb_pred
                Edges_in_cp = torch.max(Edges_in_cp_m , dim=1)[0] # numb_pred
                Edges_in_cp = torch.sum(Edges_in_cp, dim=0) # scalar
                Edges_in_cp = Edges_in_cp / image_neg_numb_pred[idx_batch]
                S_edges_in_cp[idx_batch] = Edges_in_cp
                Edges_cp_in = torch.max(Edges_in_cp_m , dim=0)[0] # cap numb_pred
                Edges_cp_in = torch.sum(Edges_cp_in, dim=0) # scalar
                Edges_cp_in = Edges_cp_in / caption_pos_numb_pred[idx_batch]
                S_edges_cp_in[idx_batch] = Edges_cp_in
                '''
                Edges_in_cp = torch.matmul(rels, img_pred.T) # cap_numb_pred, img_numb_pred
                Edges_cp_in = Edges_in_cp.T #torch.matmul(im_pred, rels.T)
                S_edges_in_cp[idx_batch] = torch.mean(torch.max(Edges_in_cp , dim=1)[0])
                S_edges_cp_in[idx_batch] = torch.mean(torch.max(Edges_cp_in , dim=1)[0])
                '''
            ip_count_obj += image_pos_numb_obj[idx_batch]
            ip_count_pred += image_pos_numb_pred[idx_batch]
            cp_count_pred += caption_pos_numb_pred[idx_batch]
            in_count_obj += image_neg_numb_obj[idx_batch]
            in_count_pred += image_neg_numb_pred[idx_batch]
            cn_count_pred += caption_neg_numb_pred[idx_batch]
                
        S_ip_cp = S_nodes_ip_cp * self.obj_coef + S_edges_ip_cp * self.pred_coef
        S_ip_cn = S_nodes_ip_cn * self.obj_coef + S_edges_ip_cn * self.pred_coef
        S_in_cp = S_nodes_in_cp * self.obj_coef + S_edges_in_cp * self.pred_coef
        
        S_cp_ip = S_nodes_cp_ip * self.obj_coef + S_edges_cp_ip * self.pred_coef
        S_cp_in = S_nodes_cp_in * self.obj_coef + S_edges_cp_in * self.pred_coef
        S_cn_ip = S_nodes_cn_ip * self.obj_coef + S_edges_cn_ip * self.pred_coef
       
        image_anchor = self.margin - (S_ip_cp - S_ip_cn)
        caption_anchor = self.margin - (S_ip_cp - S_in_cp)
        image_anchor = torch.clamp(image_anchor, min=0)
        caption_anchor = torch.clamp(caption_anchor, min=0)
        
        image_anchor_2 = self.margin - (S_cp_ip - S_cp_in)
        caption_anchor_2 = self.margin - (S_cp_ip - S_cn_ip)
        image_anchor_2 = torch.clamp(image_anchor_2, min=0)
        caption_anchor_2 = torch.clamp(caption_anchor_2, min=0)
        
        loss = image_anchor + caption_anchor + image_anchor_2 + caption_anchor_2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


def calculate_recall_retrieval(retrieval_result, k=[1,5,10], mode='i2t'):
    '''
    Calculate Recall@k from the retrieval_result matrix
    retrieval_result: tensor [numb_image, numb_caption] if mode=='i2t' else [numb_caption, numb_image]
    retrieval_result: sorted matching pairs of image and caption (index) based on their similarity
    k: list --> list of k value to calculate recall@k
    mode: str 'i2t' or 't2i'
    '''
    if mode == 'i2t':
        numb_img = retrieval_result.shape[0]
        numb_cap = retrieval_result.shape[1]
    else:
        numb_img = retrieval_result.shape[1]
        numb_cap = retrieval_result.shape[0]
    assert numb_img <= numb_cap
    
    result = []
    
    for k_value in k:
        if mode == 'i2t':
            count = 0
            for idx in range(numb_img):
                for pair_idx in range(5*idx, 5*(idx+1)):
                    if pair_idx in retrieval_result[idx,0:k_value]:
                        count += 1
                        break
            acc = count / numb_img
            result.append(acc)
        else:
            count = 0
            for idx in range(numb_cap):
                if int(np.floor(idx/5)) in retrieval_result[idx,0:k_value]:
                    count += 1
            acc = count / numb_cap
            result.append(acc)
    return result

def similarity_score_i2t(image_o, image_p, image_numb_o, image_numb_p, caption_sent, caption_rels, caption_len_sent, caption_numb_p):
    '''
    image_o: list (n_image) of tensor [numb object, dim] embeded object of all images
    image_p: list (n_image_ of tensor [numb predicates, dim] embeded predicate of all images
    image_numb_o: tensor [number of image] number of object in each image
    image_numb_p: tensor [number of image] number of predicates in each image
    caption_sent: list (n_caption) of tensor [length of sent, dim] embeded sentence of all captions
    caption_rels: list (n_caption) of tensor [total rels of all sent, dim] embeded relations of all captions
    caption_len_sent: list [number of caption] length of sentence of each caption
    caption_numb_p: tensor [number of caption] number of predicates (rels) in each caption 
    (sum(caption_numb_p) == len(caption_rels))
    given image --> find correspond caption
    each image will have 5 matching captions
    '''
    
    obj_coef = 1
    pred_coef = 1
    caption_len_sent_tensor = torch.LongTensor(caption_len_sent)
    image_o_pad = pad_sequence(image_o, batch_first=True).to(device) # n_image, max numb obj, dim
    image_p_pad = pad_sequence(image_o, batch_first=True).to(device) # n_image, max numb pred, dim
    caption_sent_pad = pad_sequence(caption_sent, batch_first=True) # n_caption, max length sent, dim
    caption_rels_pad = pad_sequence(caption_rels, batch_first=True) # n_caption, max numb caption pred, dim
    caption_sent_pad_T = torch.transpose(caption_sent_pad.contiguous(), 1, 2).to(device) # n_caption, dim, max length sent
    caption_rels_pad_T = torch.transpose(caption_rels_pad.contiguous(), 1, 2).to(device) # n_caption, dim, max numb caption pred
    numb_img = len(image_numb_o)
    numb_cap = len(caption_len_sent)
    similarities = torch.zeros(numb_img, numb_cap)
        
    for idx_img in range(numb_img):
        if image_numb_o[idx_img] == 0:
            Nodes_score_batch_mean = torch.zeros(numb_cap).to(device)
        else:
            current_img_o = image_o_pad[idx_img,:image_numb_o[idx_img],:] # numb obj, dim
            current_img_o_expand = current_img_o.repeat(numb_cap, 1, 1) # n_caption, numb obj, dim
            Nodes_score_batch = torch.bmm(current_img_o_expand, caption_sent_pad_T) # n_caption, numb obj,  max length sent
            # max by row (find match word in caps)
            # each row is the matching score of a object/pred in image to entire word/rels in captions   
            Nodes_score_batch_max = Nodes_score_batch.max(dim=2)[0] # n_caption, numb img obj
            # Take Sum
            Nodes_score_batch_sum = Nodes_score_batch_max.sum(dim=1) # n_caption
            # Take Mean
            Nodes_score_batch_mean = Nodes_score_batch_sum / image_numb_o[idx_img]
            
        if image_numb_p[idx_img] == 0:
            Edges_score_batch_mean = torch.zeros(numb_cap).to(device)
        else:
            current_img_p = image_p_pad[idx_img,:image_numb_p[idx_img],:] # numb img pred, dim
            current_img_p_expand = current_img_p.repeat(numb_cap, 1, 1) # n_caption, numb img pred, dim
            Edges_score_batch = torch.bmm(current_img_p_expand, caption_rels_pad_T) # n_caption, numb img pred,  max numb cap pred
            # max by row (find match word in caps)
            # each row is the matching score of a object/pred in image to entire word/rels in captions
            Edges_score_batch_max = Edges_score_batch.max(dim=2)[0] # n_caption, numb img pred
            # Take Sum
            Edges_score_batch_sum = Edges_score_batch_max.sum(dim=1) # n_caption
            # Take Mean
            Edges_score_batch_mean = Edges_score_batch_sum / image_numb_p[idx_img]

        # Sim score
        Sim_score = obj_coef*Nodes_score_batch_mean + pred_coef*Edges_score_batch_mean
        similarities[idx_img] = Sim_score.data.cpu()
    
    return similarities

def retrieval_i2t(sims):
    # sims tensor (numb_img, numb_caps) similarity scores
    numb_img = sims.shape[0]
    numb_cap = sims.shape[1]
    retrieval_result = torch.zeros((numb_img, numb_cap), dtype=int)
    for idx_img in range(numb_img):
        ranked_idx_cap = torch.sort(sims[idx_img],descending=True)[1]
        retrieval_result[idx_img] = ranked_idx_cap
    return retrieval_result
        
def similarity_score_t2i(image_o, image_p, image_numb_o, image_numb_p, caption_sent, caption_rels, caption_len_sent, caption_numb_p):
    '''
    image_o: list (n_image) of tensor [numb object, dim] embeded object of all images
    image_p: list (n_image_ of tensor [numb predicates, dim] embeded predicate of all images
    image_numb_o: tensor [number of image] number of object in each image
    image_numb_p: tensor [number of image] number of predicates in each image
    caption_sent: list (n_caption) of tensor [length of sent, dim] embeded sentence of all captions
    caption_rels: list (n_caption) of tensor [total rels of all sent, dim] embeded relations of all captions
    caption_len_sent: list [number of caption] length of sentence of each caption
    caption_numb_p: tensor [number of caption] number of predicates (rels) in each caption 
    (sum(caption_numb_p) == len(caption_rels))
    given image --> find correspond caption
    each image will have 5 matching captions
    '''
    
    obj_coef = 1
    pred_coef = 1
    caption_len_sent_tensor = torch.LongTensor(caption_len_sent)
    caption_len_sent_tensor[caption_len_sent_tensor==0] = 1
    image_o_pad = pad_sequence(image_o, batch_first=True).to(device) # n_image, max numb obj, dim
    image_p_pad = pad_sequence(image_o, batch_first=True).to(device) # n_image, max numb pred, dim
    caption_sent_pad = pad_sequence(caption_sent, batch_first=True) # n_caption, max length sent, dim
    caption_rels_pad = pad_sequence(caption_rels, batch_first=True) # n_caption, max numb caption pred, dim
    caption_sent_pad_T = torch.transpose(caption_sent_pad.contiguous(), 1, 2).to(device) # n_caption, dim, max length sent
    caption_rels_pad_T = torch.transpose(caption_rels_pad.contiguous(), 1, 2).to(device) # n_caption, dim, max numb caption pred
    numb_img = len(image_numb_o)
    numb_cap = len(caption_len_sent)
    similarities = torch.zeros(numb_cap, numb_img)
    
    for idx_cap in range(numb_cap):
        if caption_len_sent[idx_cap] == 0:
            Nodes_score_batch_mean = torch.zeros(numb_img).to(device)
        else:
            current_caption_sent_pad_T = caption_sent_pad_T[idx_cap,:,:caption_len_sent[idx_cap]] # dim, len sent
            current_caption_sent_pad_T_expand = current_caption_sent_pad_T.repeat(numb_img, 1, 1) # n_image, dim, len sent
            Nodes_score_batch = torch.bmm(image_o_pad, current_caption_sent_pad_T_expand) # n_img, numb img obj, len sent
            # max by column (find match word in image)
            # each row is the matching score of a object/pred in image to entire word/rels in captions   
            Nodes_score_batch_max = Nodes_score_batch.max(dim=1)[0] # n_img, len sent
            # Take Sum
            Nodes_score_batch_sum = Nodes_score_batch_max.sum(dim=1) # n_img
            # Take Mean
            Nodes_score_batch_mean = Nodes_score_batch_sum / caption_len_sent[idx_cap] # n_img
            
        if caption_numb_p[idx_cap] == 0:
            Edges_score_batch_mean = torch.zeros(numb_img).to(device)
        else:
            current_caption_rels_pad_T = caption_rels_pad_T[idx_cap,:,:caption_numb_p[idx_cap]] # dim, cap numb_p
            current_caption_rels_pad_T_expand = current_caption_rels_pad_T.repeat(numb_img, 1, 1) # n_image, dim, cap numb_p
            Edges_score_batch = torch.bmm(image_p_pad, current_caption_rels_pad_T_expand) # n_img, numb img p, numb cap p
            # max by column (find match word in image)
            # each row is the matching score of a object/pred in image to entire word/rels in captions   
            Edges_score_batch_max = Edges_score_batch.max(dim=1)[0] # n_img, numb cap p
            # Take Sum
            Edges_score_batch_sum = Edges_score_batch_max.sum(dim=1) # n_img
            # Take Mean
            Edges_score_batch_mean = Edges_score_batch_sum / caption_numb_p[idx_cap] # n_img
            
        # Sim score
        Sim_score = obj_coef*Nodes_score_batch_mean + pred_coef*Edges_score_batch_mean
        similarities[idx_cap] = Sim_score.data.cpu()
    
    return similarities

def retrieval_t2i(sims):
    # sims tensor (numb_caps, numb_img) similarity scores
    numb_img = sims.shape[1]
    numb_cap = sims.shape[0]
    retrieval_result = torch.zeros((numb_cap, numb_img), dtype=int)
    for idx_cap in range(numb_cap):
        ranked_idx_cap = torch.sort(sims[idx_cap],descending=True)[1]
        retrieval_result[idx_cap] = ranked_idx_cap
    return retrieval_result
