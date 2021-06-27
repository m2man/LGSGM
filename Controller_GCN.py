# Precompute EfficientNet before, not run EfficientNet here
# Include the Predicate visual Ft
# Add Extra GCN for textual graph (after the RNN)
from data_utils import *
import models as md
from metrics import *
from retrieval_utils import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import itertools
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import time
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os
import torchvision.transforms as transforms

device = torch.device('cuda:0')
non_blocking = True
# device = torch.cuda.set_device(0)

ROOT_DIR = '/home/nmduy/Graphs/GraSim'
#ROOT_DIR = '/home/hong01/Research/GraSim'
# DATA_DIR = f'{ROOT_DIR}/Essential_Data'

# VG_SGG_DICTS = f'{ROOT_DIR}/Essential_Data/VG-SGG-dicts-with-attri.json' # ENCODED DICTIONARY FROM VISUAL GENOME DATASET (PROVIDED IN THE ORIGINAL REPO OF KAIHUATANG)
# with open(VG_SGG_DICTS) as f:
#     info_dict = json.load(f)
    
DATA_DIR = '/home/nmduy/Graphs/GraSim/OriginalData'

#word2idx_cap = joblib.load(f"../NewData/flickr30k_caps_word2idx.joblib") # This dictionary include the above
word2idx_cap = joblib.load(f"../OriginalData/flickr30k_lowered_caps_word2idx_train_val.joblib")
word2idx_img_obj = joblib.load(f"../OriginalData/flickr30k_lowered_img_obj_word2idx.joblib") 
word2idx_img_pred = joblib.load(f"../OriginalData/flickr30k_lowered_img_pred_word2idx.joblib") 


TOTAL_CAP_WORDS = len(word2idx_cap)
TOTAL_IMG_OBJ = len(word2idx_img_obj)
TOTAL_IMG_PRED = len(word2idx_img_pred)

# lemmatized or lowered
subset = 'train'
images_data_train = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data.joblib")
caps_data_train = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data.joblib")
img_cap_matching_idx_train = joblib.load(f"{DATA_DIR}/image_caption_matching_flickr_{subset}.joblib")
cap_img_matching_idx_train = joblib.load(f"{DATA_DIR}/caption_image_matching_flickr_{subset}.joblib")

subset = 'val'
images_data_val = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data.joblib")
caps_data_val = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data.joblib")
img_cap_matching_idx_val = joblib.load(f"{DATA_DIR}/image_caption_matching_flickr_{subset}.joblib")
cap_img_matching_idx_val = joblib.load(f"{DATA_DIR}/caption_image_matching_flickr_{subset}.joblib")

init_embed_model_weight_cap = joblib.load(f'{DATA_DIR}/init_glove_embedding_weight_lowered_train_val.joblib')
init_embed_model_weight_cap = torch.FloatTensor(init_embed_model_weight_cap)
init_embed_model_weight_img_obj = joblib.load(f'{DATA_DIR}/init_glove_embedding_weight_lowered_img_obj.joblib')
init_embed_model_weight_img_obj = torch.FloatTensor(init_embed_model_weight_img_obj)
init_embed_model_weight_img_pred = joblib.load(f'{DATA_DIR}/init_glove_embedding_weight_lowered_img_pred.joblib')
init_embed_model_weight_img_pred = torch.FloatTensor(init_embed_model_weight_img_pred)

def print_dict(di):
    result = ''
    for key, val in di.items():
        key_upper = key.upper()
        result += f"{key_upper}: {val}\n"
    return result
    
class Trainer():
    def __init__(self, info_dict):
        super(Trainer, self).__init__()
        ##### INIT #####
        self.info_dict = info_dict
        self.numb_sample = info_dict['numb_sample'] # 50000 - number of training sample in 1 epoch
        self.numb_epoch = info_dict['numb_epoch'] # 10 - number of epoch
        self.unit_dim = 300
        self.numb_gcn_layers = info_dict['numb_gcn_layers'] # number of gin layer in graph embedding
        self.gcn_hidden_dim = info_dict['gcn_hidden_dim'] # hidden layer in each gin layer
        self.gcn_output_dim = info_dict['gcn_output_dim']
        self.gcn_input_dim = info_dict['gcn_input_dim']
        self.activate_fn = info_dict['activate_fn']
        self.grad_clip = info_dict['grad_clip']
        self.use_residual = False # info_dict['use_residual']    
        self.batchnorm = info_dict['batchnorm']
        self.dropout = info_dict['dropout']
        self.batch_size = info_dict['batch_size']
        self.save_dir = info_dict['save_dir']
        self.optimizer_choice = info_dict['optimizer']
        self.learning_rate = info_dict['learning_rate']
        self.model_name = info_dict['model_name']
        self.checkpoint = info_dict['checkpoint']
        self.margin_matrix_loss = info_dict['margin_matrix_loss']
        self.rnn_numb_layers = info_dict['rnn_numb_layers']
        self.rnn_bidirectional = info_dict['rnn_bidirectional']
        self.rnn_structure = info_dict['rnn_structure']
        self.visual_backbone = info_dict['visual_backbone']
        self.visual_ft_dim = info_dict['visual_ft_dim']
        self.ge_dim = info_dict['graph_emb_dim']
        self.include_pred_ft = info_dict['include_pred_ft']
        self.freeze = info_dict['freeze']
        self.datatrain = PairGraphPrecomputeDataset(image_sgg=images_data_train, caption_sgg=caps_data_train, 
                                                    word2idx_cap=word2idx_cap, word2idx_img_obj=word2idx_img_obj, 
                                                    word2idx_img_pred=word2idx_img_pred, effnet=self.visual_backbone,
                                                     image_caption_matching=img_cap_matching_idx_train, 
                                                     caption_image_matching=cap_img_matching_idx_train,
                                                     numb_sample=self.numb_sample)
        
        ## DECLARE MODEL
        self.image_branch_model = md.ImageModel(word_unit_dim=self.unit_dim, gcn_output_dim=self.gcn_output_dim, 
                                          gcn_hidden_dim=self.gcn_hidden_dim, numb_gcn_layers=self.numb_gcn_layers, 
                                          batchnorm=self.batchnorm, dropout=self.dropout, activate_fn=self.activate_fn,
                                          visualft_structure=self.visual_backbone, 
                                          visualft_feature_dim=self.visual_ft_dim, 
                                          fusion_output_dim=self.gcn_input_dim, 
                                          numb_total_obj=TOTAL_IMG_OBJ, numb_total_pred=TOTAL_IMG_PRED, 
                                          init_weight_obj=init_embed_model_weight_img_obj, 
                                          init_weight_pred=init_embed_model_weight_img_pred,
                                          include_pred_ft=self.include_pred_ft)
        
        self.gcn_model_cap = md.GCN_Network(gcn_input_dim=self.gcn_output_dim, gcn_pred_dim=self.gcn_output_dim, 
                                            gcn_output_dim=self.gcn_output_dim, gcn_hidden_dim=self.gcn_hidden_dim, 
                                            numb_gcn_layers=self.numb_gcn_layers, batchnorm=self.batchnorm, 
                                            dropout=self.dropout,activate_fn=self.activate_fn, use_residual=False)
        
        self.embed_model_cap = md.WordEmbedding(numb_words=TOTAL_CAP_WORDS, embed_dim=self.unit_dim, 
                                                init_weight=init_embed_model_weight_cap, sparse=False)

        self.sent_model = md.SentenceModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim, 
                                           numb_layers=self.rnn_numb_layers, 
                                           dropout=self.dropout, bidirectional=self.rnn_bidirectional, 
                                           structure=self.rnn_structure)

        self.rels_model = md.RelsModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim, 
                                       numb_layers=self.rnn_numb_layers, 
                                       dropout=self.dropout, bidirectional=self.rnn_bidirectional, 
                                       structure=self.rnn_structure)
        
        self.graph_embed_model = md.GraphEmb(node_dim=self.gcn_output_dim, edge_dim=self.gcn_output_dim,
                                             fusion_dim=self.ge_dim, activate_fn=self.activate_fn, 
                                             batchnorm=self.batchnorm, dropout=self.dropout)
        
        self.embed_model_cap = self.embed_model_cap.to(device)
        self.image_branch_model = self.image_branch_model.to(device)
        self.sent_model = self.sent_model.to(device)
        self.rels_model = self.rels_model.to(device)
        self.gcn_model_cap = self.gcn_model_cap.to(device)
        self.graph_embed_model = self.graph_embed_model.to(device)
        
        if self.freeze: # freeze most of component
            for p in self.image_branch_model.parameters():
                p.requires_grad = False
            for p in self.embed_model_cap.parameters():
                p.requires_grad = False
            for p in self.sent_model.parameters():
                p.requires_grad = False
            for p in self.rels_model.parameters():
                p.requires_grad = False
                
        ## PARAMS & OPTIMIZER
        self.params = []
        self.params += list(filter(lambda p: p.requires_grad, self.image_branch_model.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.embed_model_cap.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.sent_model.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.rels_model.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.gcn_model_cap.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.graph_embed_model.parameters()))
        
        if self.optimizer_choice.lower() == 'adam':                                                     
            self.optimizer = optim.Adam(self.params,
                                         lr=self.learning_rate,
                                         betas=(0.9, 0.999),
                                         eps=1e-08,
                                         weight_decay=0)
                                      
        if self.optimizer_choice.lower() == 'sgd':
            self.optimizer = optim.SGD(self.params,
                                        lr=self.learning_rate,
                                        momentum=0.9,
                                        weight_decay=0)
            
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 30 epochs"""
        lr = self.learning_rate * (0.1 ** (epoch // 15)) # 15 epoch update once
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        
    # ---------- WRITE INFO TO TXT FILE ---------
    def extract_info(self):
        try:
            timestampLaunch = self.timestampLaunch
        except:
            timestampLaunch = 'undefined'
        model_info_log = open(f"{self.save_dir}/{self.model_name}-{timestampLaunch}-INFO.log", "w")
        result = f"===== {self.model_name} =====\n"
        result += print_dict(self.info_dict)
        model_info_log.write(result)
        model_info_log.close()
        
    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        #---- Load checkpoint 
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.embed_model_cap.load_state_dict(modelCheckpoint['embed_model_cap_state_dict'])
            self.sent_model.load_state_dict(modelCheckpoint['sent_model_state_dict'])
            self.image_branch_model.load_state_dict(modelCheckpoint['image_branch_model_state_dict'])
            self.rels_model.load_state_dict(modelCheckpoint['rels_model_state_dict'])
            self.gcn_model_cap.load_state_dict(modelCheckpoint['gcn_model_cap_state_dict'])
            self.graph_embed_model.load_state_dict(modelCheckpoint['graph_embed_model_state_dict'])
            if not self.freeze:
                self.optimizer.load_state_dict(modelCheckpoint['optimizer_state_dict'])
        else:
            print("TRAIN FROM SCRATCH")
            
            
    # ---------- RUN TRAIN ---------
    def train(self):
        ## LOAD PRETRAINED MODEL ##
        self.load_trained_model()
        
        scheduler = ReduceLROnPlateau(self.optimizer, factor = 0.2, patience=5, 
                                      mode = 'min', verbose=True, min_lr=1e-6)
        #scheduler_remaining_models = ReduceLROnPlateau(self.optimizer_remaining_models, factor = 0.5, patience=10, 
                                                  #mode = 'min', verbose=True, min_lr=1e-6)
        
        ## LOSS FUNCTION ##
        loss_matrix = ContrastiveLoss(margin=self.margin_matrix_loss, predicate_score_rate=1, 
                                      max_violation=True, cross_attn='i2t')
        loss_geb = ContrastiveLoss_CosineSimilarity(margin=self.margin_matrix_loss, max_violation=True)
        loss_matrix = loss_matrix.to(device)
        loss_geb = loss_geb.to(device)

        ## REPORT ##
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        self.timestampLaunch = timestampDate + '-' + timestampTime
        # f_log = open(f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}-REPORT.log", "w")
        writer = SummaryWriter(f'{self.save_dir}/{self.model_name}-{self.timestampLaunch}/')
        self.extract_info()
        
        ## TRAIN THE NETWORK ##
        lossMIN = 100000
        flag = 0
        count_change_loss = 0
        
        for epochID in range (self.numb_epoch):
            print(f"Training {epochID}/{self.numb_epoch-1}")

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            
            # Update learning rate at each epoch
            #self.adjust_learning_rate(epochID)
            
            lossTrain, lossTrainMatrix, lossTrainGEB = self.train_epoch(loss_matrix, loss_geb, writer, epochID)
            # lossVal = self.val_epoch(epochID, loss_matrix)
            with torch.no_grad():
                if self.freeze:
                    lossVal, ar_val, ari_val = self.validate_retrieval(images_data_val, caps_data_val, include_geb=True)
                else:
                    lossVal, ar_val, ari_val = self.validate_retrieval(images_data_val, caps_data_val, include_geb=False)
                #lossTrain_recall = self.validate_retrieval(images_data_train, caps_data_train)
            lossVal = 6 - lossVal # 6 recall overall
            #lossTrain_recall = 6 - lossTrain_recall
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(lossVal)
            #scheduler_remaining_models.step(lossVal)
            info_txt = f"Epoch {epochID + 1}/{self.numb_epoch} [{timestampEND}]"
            
            if lossVal < lossMIN:
                count_change_loss = 0
                if lossVal < lossMIN:
                    lossMIN = lossVal
                torch.save({'epoch': epochID, \
                            'embed_model_cap_state_dict': self.embed_model_cap.state_dict(),
                            'sent_model_state_dict': self.sent_model.state_dict(),
                            'image_branch_model_state_dict': self.image_branch_model.state_dict(),
                            'rels_model_state_dict': self.rels_model.state_dict(),
                            'gcn_model_cap_state_dict': self.gcn_model_cap.state_dict(),
                            'graph_embed_model_state_dict': self.graph_embed_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_loss': lossMIN}, f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}.pth.tar")
                info_txt = info_txt + f" [SAVE]\nLoss Val: {lossVal}"
               
            else:
                count_change_loss += 1
                info_txt = info_txt + f"\nLoss Val: {lossVal}"   
            print(info_txt)
            info_txt = info_txt + f"\n[i2t] {round(ar_val[0], 4)} {round(ar_val[1], 4)} {round(ar_val[2], 4)}"
            info_txt = info_txt + f"\n[t2i] {round(ari_val[0], 4)} {round(ari_val[1], 4)} {round(ari_val[2], 4)}"
            info_txt = info_txt + f"\nLoss Train: {round(lossTrain,6)} -- Loss Train Matrix: {round(lossTrainMatrix,6)} -- Loss Train GEB: {round(lossTrainGEB,6)}\n----------\n"
            
            with open(f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}-REPORT.log", "a") as f_log:
                f_log.write(info_txt)
                    
            writer.add_scalars('Loss Epoch', {'train': lossTrain}, epochID)
            writer.add_scalars('Loss Epoch', {'trainmatrix': lossTrainMatrix}, epochID)
            writer.add_scalars('Loss Epoch', {'traingeb': lossTrainGEB}, epochID)
            writer.add_scalars('Loss Epoch', {'val': lossVal}, epochID)
            writer.add_scalars('Loss Epoch', {'val-best': lossMIN}, epochID)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, epochID)

            if count_change_loss >= 15:
                print(f'Early stopping: {count_change_loss} epoch not decrease the loss')
                break

        # f_log.close()
        writer.close()
    
    # ---------- TRAINING 1 EPOCH ---------
    def train_epoch(self, loss_matrix, loss_geb, writer, epochID):
        print(f"Shuffling Training Dataset")
        self.datatrain.create_pairs(seed=1509+epochID+100)
        print(f"Done Shuffling")
        '''
        numb_sample = len(self.datatrain)
        temp = [x['label'] for x in self.datatrain]
        numb_match = np.sum(np.asarray(temp))
        numb_unmatch = numb_sample - numb_match
        print(f"Total Training sample: {numb_sample} --- Matched sample: {numb_match} --- UnMatched sample: {numb_unmatch}")
        '''
        
        dataloadertrain = make_PairGraphPrecomputeDataLoader(self.datatrain, batch_size=self.batch_size, num_workers=0)
        
        self.embed_model_cap.train()
        self.image_branch_model.train()
        self.sent_model.train()
        self.rels_model.train()
        self.gcn_model_cap.train()
        self.graph_embed_model.train()
        
        loss_report = 0
        loss_matrix_report = 0
        loss_geb_report = 0
        count = 0
        numb_iter = len(dataloadertrain)
        print(f"Total iteration: {numb_iter}")
        for batchID, batch in enumerate(dataloadertrain):
            ## CUDA ##
            #if batchID > 2:
            #    break
            img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
            cap_p_o, cap_p_p, cap_p_e, cap_p_s, cap_p_numb_o, cap_p_numb_p, cap_p_len_p, cap_p_len_s = batch
            
            batch_size = len(cap_p_len_s)
            
            img_p_o_ft = img_p_o_ft.to(device)
            img_p_p_ft = img_p_p_ft.to(device)
            img_p_o = img_p_o.to(device)
            img_p_p = img_p_p.to(device) 
            img_p_e = img_p_e.to(device)
            # cap_p_o = cap_p_o.to(device)
            # cap_p_p = cap_p_p.to(device) 
            cap_p_e = cap_p_e.to(device)
            
            if not self.include_pred_ft:
                img_p_p_ft = None
                
            ## [Image] GCN Network
            gcn_eb_img_p_o, gcn_eb_img_p_p = self.image_branch_model(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p, img_p_e)
            
            # [Caption] Padding
            pad_cap_p_s = pad_sequence(cap_p_s, batch_first=True) # padding sentence
            pad_cap_p_p = pad_sequence(cap_p_p, batch_first=True) # padding predicates
            
            ## Embedding network (object, predicates in image and caption)
            # [Caption] Embed Sentence and Predicates
            eb_pad_cap_p_s = self.embed_model_cap(pad_cap_p_s.to(device))
            eb_pad_cap_p_p = self.embed_model_cap(pad_cap_p_p.to(device))
                
            ## [Caption] Sentence Model
            rnn_eb_pad_cap_p_s = self.sent_model(eb_pad_cap_p_s, cap_p_len_s) # ncap, max sent len, dim

            # Concate for batch processing    
            rnn_eb_cap_p_rels, rnn_eb_cap_p_rels_nodes = self.rels_model(eb_pad_cap_p_p, cap_p_len_p) # total rels, dim
            
            # [CAPTION] GCN for object and edge in relations
            total_cap_p_numb_o = sum(cap_p_numb_o)
            total_cap_p_numb_p = sum(cap_p_numb_p)
            eb_cap_p_o = torch.zeros(total_cap_p_numb_o, self.gcn_output_dim).to(device)
            eb_cap_p_p = torch.zeros(total_cap_p_numb_p, self.gcn_output_dim).to(device)
            for idx in range(len(rnn_eb_cap_p_rels_nodes)):
                edge = cap_p_e[idx] # subject, object
                eb_cap_p_o[edge[0]] = rnn_eb_cap_p_rels_nodes[idx,1,:] # <start> is 1st token
                eb_cap_p_o[edge[1]] = rnn_eb_cap_p_rels_nodes[idx,cap_p_len_p[idx]-2 ,:] # <end> is last token
                eb_cap_p_p[idx] = torch.mean(rnn_eb_cap_p_rels_nodes[idx,2:(cap_p_len_p[idx]-2),:], dim=0)
                #if cap_p_len_p[idx] > 5: # pred is longer than 1 words
                #    eb_cap_p_p[idx] = torch.mean(rnn_eb_cap_p_rels_nodes[idx,2:(cap_p_len_p[idx]-2),:], dim=0)
                #else:
                #    eb_cap_p_p[idx] = rnn_eb_cap_p_rels_nodes[idx,2,:]
            eb_cap_p_o, eb_cap_p_p = self.gcn_model_cap(eb_cap_p_o, eb_cap_p_p, cap_p_e)
            
            # [GRAPHEMB]
            image_geb = self.graph_embed_model(gcn_eb_img_p_o, gcn_eb_img_p_p, img_p_numb_o, img_p_numb_p) # n_img, dim
            caption_geb = self.graph_embed_model(eb_cap_p_o, eb_cap_p_p, cap_p_numb_o, cap_p_numb_p) # n_cap, dim         
                
            ## LOSS
            # Assign to right format
            # img_obj [batch_size, max obj, dim], img_pred [batch_size, max pred, dim]
            max_obj_n, max_pred_n, max_rels_n = max(img_p_numb_o), max(img_p_numb_p), max(cap_p_numb_p)
            img_emb = torch.zeros(len(img_p_numb_o), max_obj_n, gcn_eb_img_p_o.shape[1]).to(device)
            pred_emb = torch.zeros(len(img_p_numb_p), max_pred_n, gcn_eb_img_p_p.shape[1]).to(device)
            obj_offset = 0
            for i, obj_num in enumerate(img_p_numb_o):
                img_emb[i][:obj_num,:] = gcn_eb_img_p_o[obj_offset:obj_offset+obj_num,:]
                obj_offset+= obj_num
            pred_offset = 0
            for i, pred_num in enumerate(img_p_numb_p):
                pred_emb[i][:pred_num,:] = gcn_eb_img_p_p[pred_offset : pred_offset + pred_num,:]
                pred_offset+= pred_num
                
            rels_emb = torch.zeros(len(cap_p_numb_p), max_rels_n, rnn_eb_cap_p_rels.shape[1]).to(device)
            rels_offset = 0
            for i, rels_num in enumerate(cap_p_numb_p):
                rels_emb[i][:rels_num,:] = rnn_eb_cap_p_rels[rels_offset : rels_offset + rels_num,:]
                rels_offset+= rels_num
                
            lossvalue_matrix = loss_matrix(img_emb, img_p_numb_o, rnn_eb_pad_cap_p_s, cap_p_len_s, 
                                            pred_emb, img_p_numb_p, rels_emb, cap_p_numb_p)
            lossvalue_geb = loss_geb(image_geb, caption_geb)
            lossvalue = lossvalue_matrix + lossvalue_geb
            
            ## Update ##
            self.optimizer.zero_grad()
            lossvalue.backward()
            if self.grad_clip > 0:
                clip_grad_norm(self.params,
                               self.grad_clip)
            self.optimizer.step()
            loss_report += lossvalue.item()
            loss_matrix_report += lossvalue_matrix.item()
            loss_geb_report += lossvalue_geb.item()
            count += 1
            if (batchID+1) % 300 == 0:
                print(f"Batch Idx: {batchID+1} / {len(dataloadertrain)}: Loss Train {round(loss_report/count, 6)} -- Loss Matrix {round(loss_matrix_report/count, 6)} -- Loss GEB {round(loss_geb_report/count, 6)}")
                writer.add_scalars('Loss Training Iter', {'loss': loss_report/count}, epochID * np.floor(numb_iter/20) + np.floor((batchID+1)/20))
                writer.add_scalars('Loss Training Iter', {'loss': loss_matrix_report/count}, epochID * np.floor(numb_iter/20) + np.floor((batchID+1)/20))
                writer.add_scalars('Loss Training Iter', {'loss': loss_geb_report/count}, epochID * np.floor(numb_iter/20) + np.floor((batchID+1)/20))
                
        return loss_report/count, loss_matrix_report/count, loss_geb_report/count
    
    def encode_image_sgg(self, image_sgg, batch_size=16):
        i_dts = ImagePrecomputeDataset(image_sgg=image_sgg, word2idx_obj=word2idx_img_obj, 
                                       word2idx_pred=word2idx_img_pred, effnet=self.visual_backbone, numb_sample=None)
        i_dtld = make_ImagePrecomputeDataLoader(i_dts, batch_size=batch_size, num_workers=0)
        
        eb_img_o_all = []
        eb_img_p_all = []
        img_numb_o_all = []
        img_numb_p_all = []
        image_geb_all = []
        
        self.image_branch_model.eval()
        self.graph_embed_model.eval()
        
        with torch.no_grad():
            print('Embedding objects and predicates of images ...')
            for batchID, batch in enumerate(i_dtld):
                img_o, img_o_ft, img_p, img_p_ft, img_e, img_numb_o, img_numb_p = batch
                    
                img_o_ft = img_o_ft.to(device)
                img_p_ft = img_p_ft.to(device)
                img_o = img_o.to(device)
                img_p = img_p.to(device) 
                img_e = img_e.to(device)
                
                if not self.include_pred_ft:
                    img_p_ft = None
                ## [Image] GCN Network
                gcn_eb_img_o, gcn_eb_img_p = self.image_branch_model(img_o_ft, img_p_ft, img_o, img_p, img_e)
                
                obj_offset = 0
                pred_offset = 0
                for idx_img in range(len(img_numb_o)):
                    eb_img_o_all.append(gcn_eb_img_o[obj_offset: (obj_offset + img_numb_o[idx_img]),:].data.cpu())
                    eb_img_p_all.append(gcn_eb_img_p[pred_offset: (pred_offset + img_numb_p[idx_img]),:].data.cpu())
                    obj_offset += img_numb_o[idx_img]
                    pred_offset += img_numb_p[idx_img]
                img_numb_o_all += img_numb_o
                img_numb_p_all += img_numb_p
                image_geb = self.graph_embed_model(gcn_eb_img_o, gcn_eb_img_p, img_numb_o, img_numb_p) # n_img, dim
                image_geb_all.append(image_geb)
                
            image_geb_all = torch.cat(image_geb_all, dim=0).data.cpu().numpy()
            img_obj_emb = pad_sequence(eb_img_o_all, batch_first=True).data.cpu().numpy() # nimg, max obj, dim
            img_pred_emb = pad_sequence(eb_img_p_all, batch_first=True).data.cpu().numpy() # nimg, max pred, dim
            del img_o, img_p, img_e
        return img_obj_emb, img_pred_emb, img_numb_o_all, img_numb_p_all, image_geb_all
    
    def encode_caption_sgg(self, caption_sgg, batch_size=1):   
        c_dts = CaptionDataset(caption_sgg=caption_sgg, word2idx=word2idx_cap, numb_sample=None)
        c_dtld = make_CaptionDataLoader(c_dts, batch_size=batch_size, num_workers=0)
        
        eb_cap_rels_all = []
        eb_cap_sent_all = []        
        cap_numb_rels_all = []
        cap_len_sent_all = []
        caption_geb_all = []
        
        self.embed_model_cap.eval()
        self.gcn_model_cap.eval()
        self.sent_model.eval()
        self.rels_model.eval()
        self.graph_embed_model.eval()
        
        with torch.no_grad():
            print('Embedding captions data ...')
            for batchID, batch in enumerate(c_dtld):
                cap_o, cap_p, cap_e, cap_s, cap_numb_o, cap_numb_p, cap_len_p, cap_len_s = batch
                batch_size = len(cap_numb_o)
  
                pad_cap_s_concate = pad_sequence(cap_s, batch_first=True).to(device) # padding Sentence
                pad_cap_p_concate = pad_sequence(cap_p, batch_first=True).to(device) # padding Rels

                ## Embedding network (object, predicates in image and caption)
                # [Caption] Embed Sentence
                eb_pad_cap_s_concate = self.embed_model_cap(pad_cap_s_concate)
                eb_pad_cap_p_concate = self.embed_model_cap(pad_cap_p_concate)
                
                ## [Caption] Sentence Model
                rnn_eb_pad_cap_s_concate = self.sent_model(eb_pad_cap_s_concate, cap_len_s)
                for idx_sent in range(len(cap_len_s)):
                    eb_cap_sent_all.append(rnn_eb_pad_cap_s_concate[idx_sent, 0:cap_len_s[idx_sent], :].data.cpu())
                    
                rnn_eb_cap_rels, rnn_eb_cap_rels_nodes = self.rels_model(eb_pad_cap_p_concate, cap_len_p)
                
                # [CAPTION] GCN for object and edge in relations
                total_cap_numb_o = sum(cap_numb_o)
                total_cap_numb_p = sum(cap_numb_p)
                eb_cap_o = torch.zeros(total_cap_numb_o, self.gcn_output_dim).to(device)
                eb_cap_p = torch.zeros(total_cap_numb_p, self.gcn_output_dim).to(device)
                for idx in range(len(rnn_eb_cap_rels_nodes)):
                    edge = cap_e[idx] # subject, object
                    eb_cap_o[edge[0]] = rnn_eb_cap_rels_nodes[idx,1,:] # <start> is 1st token
                    eb_cap_o[edge[1]] = rnn_eb_cap_rels_nodes[idx,cap_len_p[idx]-2 ,:] # <end> is last token
                    eb_cap_p[idx] = torch.mean(rnn_eb_cap_rels_nodes[idx,2:(cap_len_p[idx]-2),:], dim=0)
                    #if cap_p_len_p[idx] > 5: # pred is longer than 1 words
                    #    eb_cap_p_p[idx] = torch.mean(rnn_eb_cap_p_rels_nodes[idx,2:(cap_p_len_p[idx]-2),:], dim=0)
                    #else:
                    #    eb_cap_p_p[idx] = rnn_eb_cap_p_rels_nodes[idx,2,:]
                eb_cap_o, eb_cap_p = self.gcn_model_cap(eb_cap_o, eb_cap_p, cap_e)

                # [GRAPHEMB]
                caption_geb = self.graph_embed_model(eb_cap_o, eb_cap_p, cap_numb_o, cap_numb_p) # n_cap, dim  
                caption_geb_all.append(caption_geb)
                
                pred_offset = 0
                for idx_cap in range(len(cap_numb_p)):
                    eb_cap_rels_all.append(rnn_eb_cap_rels[pred_offset : (pred_offset+cap_numb_p[idx_cap]),:].data.cpu())
                    pred_offset += cap_numb_p[idx_cap]
                    
                cap_numb_rels_all += cap_numb_p
                cap_len_sent_all += cap_len_s # cap_len_s is a list already # list [number of caption]
            
            caption_geb_all = torch.cat(caption_geb_all, dim=0).data.cpu().numpy()
            cap_sent_emb = pad_sequence(eb_cap_sent_all, batch_first=True).data.cpu().numpy() # ncaption, max len, dim
            cap_rels_emb = pad_sequence(eb_cap_rels_all, batch_first=True).data.cpu().numpy() # ncaption, max rels, dim
            
        return cap_sent_emb, cap_rels_emb, cap_len_sent_all, cap_numb_rels_all, caption_geb_all
    
    # ---------- VALIDATE ---------
    def validate_retrieval(self, image_sgg, caption_sgg, include_geb=False):
        print('---------- VALIDATE RETRIEVAL ----------')
        
        img_obj_emb, img_pred_emb, img_numb_o_all, img_numb_p_all, img_geb_all = self.encode_image_sgg(image_sgg, batch_size=16)
        cap_sent_emb, cap_rels_emb, cap_len_sent_all, cap_numb_rels_all, cap_geb_all = self.encode_caption_sgg(caption_sgg, batch_size=64)
        
        if not include_geb:
            img_geb_all = None
            cap_geb_all = None
        print('Scoring ...')    
        # Perform retrieval
        with torch.no_grad():
            score, ar, ari = evalrank(img_obj_emb, img_numb_o_all, cap_sent_emb, cap_len_sent_all, 
                             img_pred_emb, img_numb_p_all, cap_rels_emb, cap_numb_rels_all,
                             cross_attn='i2t', predicate_score_rate=1, img_geb=img_geb_all, cap_geb=cap_geb_all)
        return score, ar, ari

# ----- EVALUATOR -----
class Evaluator():
    def __init__(self, info_dict):
        super(Evaluator, self).__init__()
        ##### INIT #####
        self.info_dict = info_dict
        self.numb_sample = info_dict['numb_sample'] # 50000 - number of training sample in 1 epoch
        self.numb_epoch = info_dict['numb_epoch'] # 10 - number of epoch
        self.unit_dim = 300
        self.numb_gcn_layers = info_dict['numb_gcn_layers'] # number of gin layer in graph embedding
        self.gcn_hidden_dim = info_dict['gcn_hidden_dim'] # hidden layer in each gin layer
        self.gcn_output_dim = info_dict['gcn_output_dim']
        self.gcn_input_dim = info_dict['gcn_input_dim']
        self.activate_fn = info_dict['activate_fn']
        self.grad_clip = info_dict['grad_clip']
        self.use_residual = False # info_dict['use_residual']    
        self.batchnorm = info_dict['batchnorm']
        self.dropout = info_dict['dropout']
        self.checkpoint = info_dict['checkpoint']
        self.margin_matrix_loss = info_dict['margin_matrix_loss']
        self.rnn_numb_layers = info_dict['rnn_numb_layers']
        self.rnn_bidirectional = info_dict['rnn_bidirectional']
        self.rnn_structure = info_dict['rnn_structure']
        self.visual_backbone = info_dict['visual_backbone']
        self.visual_ft_dim = info_dict['visual_ft_dim']
        self.ge_dim = info_dict['graph_emb_dim']
        self.include_pred_ft = info_dict['include_pred_ft']
        self.datatrain = PairGraphPrecomputeDataset(image_sgg=images_data_train, caption_sgg=caps_data_train, 
                                                    word2idx_cap=word2idx_cap, word2idx_img_obj=word2idx_img_obj, 
                                                    word2idx_img_pred=word2idx_img_pred, effnet=self.visual_backbone,
                                                     image_caption_matching=img_cap_matching_idx_train, 
                                                     caption_image_matching=cap_img_matching_idx_train,
                                                     numb_sample=self.numb_sample)
        
        ## DECLARE MODEL
        self.image_branch_model = md.ImageModel(word_unit_dim=self.unit_dim, gcn_output_dim=self.gcn_output_dim, 
                                          gcn_hidden_dim=self.gcn_hidden_dim, numb_gcn_layers=self.numb_gcn_layers, 
                                          batchnorm=self.batchnorm, dropout=self.dropout, activate_fn=self.activate_fn,
                                          visualft_structure=self.visual_backbone, 
                                          visualft_feature_dim=self.visual_ft_dim, 
                                          fusion_output_dim=self.gcn_input_dim, 
                                          numb_total_obj=TOTAL_IMG_OBJ, numb_total_pred=TOTAL_IMG_PRED, 
                                          init_weight_obj=init_embed_model_weight_img_obj, 
                                          init_weight_pred=init_embed_model_weight_img_pred,
                                          include_pred_ft=self.include_pred_ft)
        
        self.gcn_model_cap = md.GCN_Network(gcn_input_dim=self.gcn_output_dim, gcn_pred_dim=self.gcn_output_dim, 
                                            gcn_output_dim=self.gcn_output_dim, gcn_hidden_dim=self.gcn_hidden_dim, 
                                            numb_gcn_layers=self.numb_gcn_layers, batchnorm=self.batchnorm, 
                                            dropout=self.dropout,activate_fn=self.activate_fn, use_residual=False)
        
        self.embed_model_cap = md.WordEmbedding(numb_words=TOTAL_CAP_WORDS, embed_dim=self.unit_dim, 
                                                init_weight=init_embed_model_weight_cap, sparse=False)

        self.sent_model = md.SentenceModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim, 
                                           numb_layers=self.rnn_numb_layers, 
                                           dropout=self.dropout, bidirectional=self.rnn_bidirectional, 
                                           structure=self.rnn_structure)

        self.rels_model = md.RelsModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim, 
                                       numb_layers=self.rnn_numb_layers, 
                                       dropout=self.dropout, bidirectional=self.rnn_bidirectional, 
                                       structure=self.rnn_structure)
        
        self.graph_embed_model = md.GraphEmb(node_dim=self.gcn_output_dim, edge_dim=self.gcn_output_dim,
                                             fusion_dim=self.ge_dim, activate_fn=self.activate_fn, 
                                             batchnorm=self.batchnorm, dropout=self.dropout)
        
        self.embed_model_cap = self.embed_model_cap.to(device)
        self.image_branch_model = self.image_branch_model.to(device)
        self.sent_model = self.sent_model.to(device)
        self.rels_model = self.rels_model.to(device)
        self.gcn_model_cap = self.gcn_model_cap.to(device)
        self.graph_embed_model = self.graph_embed_model.to(device)
        
        self.image_branch_model.eval()
        self.embed_model_cap.eval()
        self.sent_model.eval()
        self.rels_model.eval()
        self.gcn_model_cap.eval()
        self.graph_embed_model.eval()
        
    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        #---- Load checkpoint 
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.embed_model_cap.load_state_dict(modelCheckpoint['embed_model_cap_state_dict'])
            self.sent_model.load_state_dict(modelCheckpoint['sent_model_state_dict'])
            self.image_branch_model.load_state_dict(modelCheckpoint['image_branch_model_state_dict'])
            self.rels_model.load_state_dict(modelCheckpoint['rels_model_state_dict'])
            self.gcn_model_cap.load_state_dict(modelCheckpoint['gcn_model_cap_state_dict'])
            self.graph_embed_model.load_state_dict(modelCheckpoint['graph_embed_model_state_dict'])
        else:
            print("TRAIN FROM SCRATCH")            
            
    def encode_image_sgg(self, image_sgg, batch_size=16):
        i_dts = ImagePrecomputeDataset(image_sgg=image_sgg, word2idx_obj=word2idx_img_obj, 
                                       word2idx_pred=word2idx_img_pred, effnet=self.visual_backbone, numb_sample=None)
        i_dtld = make_ImagePrecomputeDataLoader(i_dts, batch_size=batch_size, num_workers=0)
        
        eb_img_o_all = []
        eb_img_p_all = []
        img_numb_o_all = []
        img_numb_p_all = []
        image_geb_all = []
        
        self.image_branch_model.eval()
        self.graph_embed_model.eval()
        
        with torch.no_grad():
            print('Embedding objects and predicates of images ...')
            for batchID, batch in enumerate(i_dtld):
                img_o, img_o_ft, img_p, img_p_ft, img_e, img_numb_o, img_numb_p = batch
                    
                img_o_ft = img_o_ft.to(device)
                img_p_ft = img_p_ft.to(device)
                img_o = img_o.to(device)
                img_p = img_p.to(device) 
                img_e = img_e.to(device)
                
                if not self.include_pred_ft:
                    img_p_ft = None
                ## [Image] GCN Network
                gcn_eb_img_o, gcn_eb_img_p = self.image_branch_model(img_o_ft, img_p_ft, img_o, img_p, img_e)
                
                obj_offset = 0
                pred_offset = 0
                for idx_img in range(len(img_numb_o)):
                    eb_img_o_all.append(gcn_eb_img_o[obj_offset: (obj_offset + img_numb_o[idx_img]),:].data.cpu())
                    eb_img_p_all.append(gcn_eb_img_p[pred_offset: (pred_offset + img_numb_p[idx_img]),:].data.cpu())
                    obj_offset += img_numb_o[idx_img]
                    pred_offset += img_numb_p[idx_img]
                img_numb_o_all += img_numb_o
                img_numb_p_all += img_numb_p
                image_geb = self.graph_embed_model(gcn_eb_img_o, gcn_eb_img_p, img_numb_o, img_numb_p) # n_img, dim
                image_geb_all.append(image_geb)
                
            image_geb_all = torch.cat(image_geb_all, dim=0).data.cpu().numpy()
            img_obj_emb = pad_sequence(eb_img_o_all, batch_first=True).data.cpu().numpy() # nimg, max obj, dim
            img_pred_emb = pad_sequence(eb_img_p_all, batch_first=True).data.cpu().numpy() # nimg, max pred, dim
            del img_o, img_p, img_e
        return img_obj_emb, img_pred_emb, img_numb_o_all, img_numb_p_all, image_geb_all
    
    def encode_caption_sgg(self, caption_sgg, batch_size=1):   
        c_dts = CaptionDataset(caption_sgg=caption_sgg, word2idx=word2idx_cap, numb_sample=None)
        c_dtld = make_CaptionDataLoader(c_dts, batch_size=batch_size, num_workers=0)
        
        eb_cap_rels_all = []
        eb_cap_sent_all = []        
        cap_numb_rels_all = []
        cap_len_sent_all = []
        caption_geb_all = []
        
        self.embed_model_cap.eval()
        self.gcn_model_cap.eval()
        self.sent_model.eval()
        self.rels_model.eval()
        self.graph_embed_model.eval()
        
        with torch.no_grad():
            print('Embedding captions data ...')
            for batchID, batch in enumerate(c_dtld):
                cap_o, cap_p, cap_e, cap_s, cap_numb_o, cap_numb_p, cap_len_p, cap_len_s = batch
                batch_size = len(cap_numb_o)
  
                pad_cap_s_concate = pad_sequence(cap_s, batch_first=True).to(device) # padding Sentence
                pad_cap_p_concate = pad_sequence(cap_p, batch_first=True).to(device) # padding Rels

                ## Embedding network (object, predicates in image and caption)
                # [Caption] Embed Sentence
                eb_pad_cap_s_concate = self.embed_model_cap(pad_cap_s_concate)
                eb_pad_cap_p_concate = self.embed_model_cap(pad_cap_p_concate)
                
                ## [Caption] Sentence Model
                rnn_eb_pad_cap_s_concate = self.sent_model(eb_pad_cap_s_concate, cap_len_s)
                for idx_sent in range(len(cap_len_s)):
                    eb_cap_sent_all.append(rnn_eb_pad_cap_s_concate[idx_sent, 0:cap_len_s[idx_sent], :].data.cpu())
                    
                rnn_eb_cap_rels, rnn_eb_cap_rels_nodes = self.rels_model(eb_pad_cap_p_concate, cap_len_p)
                
                # [CAPTION] GCN for object and edge in relations
                total_cap_numb_o = sum(cap_numb_o)
                total_cap_numb_p = sum(cap_numb_p)
                eb_cap_o = torch.zeros(total_cap_numb_o, self.gcn_output_dim).to(device)
                eb_cap_p = torch.zeros(total_cap_numb_p, self.gcn_output_dim).to(device)
                for idx in range(len(rnn_eb_cap_rels_nodes)):
                    edge = cap_e[idx] # subject, object
                    eb_cap_o[edge[0]] = rnn_eb_cap_rels_nodes[idx,1,:] # <start> is 1st token
                    eb_cap_o[edge[1]] = rnn_eb_cap_rels_nodes[idx,cap_len_p[idx]-2 ,:] # <end> is last token
                    eb_cap_p[idx] = torch.mean(rnn_eb_cap_rels_nodes[idx,2:(cap_len_p[idx]-2),:], dim=0)
                    #if cap_p_len_p[idx] > 5: # pred is longer than 1 words
                    #    eb_cap_p_p[idx] = torch.mean(rnn_eb_cap_p_rels_nodes[idx,2:(cap_p_len_p[idx]-2),:], dim=0)
                    #else:
                    #    eb_cap_p_p[idx] = rnn_eb_cap_p_rels_nodes[idx,2,:]
                eb_cap_o, eb_cap_p = self.gcn_model_cap(eb_cap_o, eb_cap_p, cap_e)

                # [GRAPHEMB]
                caption_geb = self.graph_embed_model(eb_cap_o, eb_cap_p, cap_numb_o, cap_numb_p) # n_cap, dim  
                caption_geb_all.append(caption_geb)
                
                pred_offset = 0
                for idx_cap in range(len(cap_numb_p)):
                    eb_cap_rels_all.append(rnn_eb_cap_rels[pred_offset : (pred_offset+cap_numb_p[idx_cap]),:].data.cpu())
                    pred_offset += cap_numb_p[idx_cap]
                    
                cap_numb_rels_all += cap_numb_p
                cap_len_sent_all += cap_len_s # cap_len_s is a list already # list [number of caption]
            
            caption_geb_all = torch.cat(caption_geb_all, dim=0).data.cpu().numpy()
            cap_sent_emb = pad_sequence(eb_cap_sent_all, batch_first=True).data.cpu().numpy() # ncaption, max len, dim
            cap_rels_emb = pad_sequence(eb_cap_rels_all, batch_first=True).data.cpu().numpy() # ncaption, max rels, dim
            
        return cap_sent_emb, cap_rels_emb, cap_len_sent_all, cap_numb_rels_all, caption_geb_all
    
    # ---------- VALIDATE ---------
    def validate_retrieval(self, image_sgg, caption_sgg, include_geb=False):
        print('---------- VALIDATE RETRIEVAL ----------')
        
        img_obj_emb, img_pred_emb, img_numb_o_all, img_numb_p_all, img_geb_all = self.encode_image_sgg(image_sgg, batch_size=16)
        cap_sent_emb, cap_rels_emb, cap_len_sent_all, cap_numb_rels_all, cap_geb_all = self.encode_caption_sgg(caption_sgg, batch_size=64)
        
        if not include_geb:
            img_geb_all = None
            cap_geb_all = None
        print('Scoring ...')    
        # Perform retrieval
        with torch.no_grad():
            score, ar, ari = evalrank(img_obj_emb, img_numb_o_all, cap_sent_emb, cap_len_sent_all, 
                             img_pred_emb, img_numb_p_all, cap_rels_emb, cap_numb_rels_all,
                             cross_attn='t2i', predicate_score_rate=1, img_geb=img_geb_all, cap_geb=cap_geb_all)
        return score, ar, ari