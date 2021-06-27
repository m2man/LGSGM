# Include Pred Visual Features (optional)
import torch
import joblib
import torch.nn as nn
device = torch.device('cuda:0')
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from efficientnet_pytorch import EfficientNet

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

# MLP with batchnorm and dropout
class MLP(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=[], output_dim=300, activate_fn='swish', batchnorm=True, dropout=None, perform_at_end=True, use_residual=False):
        '''
        dropout is None or an float [0,1] --> normally 0.5
        activate_fn = 'swish' - 'relu' - 'leakyrelu'
        perform_at_end = True --> apply batchnorm, relu, dropout at the last layer (output)
        hidden_dim is a list indicating unit in hidden layers --> e.g. [1024, 512, 256]
        '''
        super(MLP, self).__init__()
        if activate_fn.lower() == 'relu':
            self.activate = nn.ReLU()
        elif activate_fn.lower() == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.2)
        elif activate_fn.lower() == 'tanh' :
            self.activate = nn.Tanh()
        else:
            self.activate = MemoryEfficientSwish()
        
        self.use_residual = use_residual
        self.perform_at_end = perform_at_end
        self.hidden_dim = hidden_dim + [output_dim]
        
        self.numb_layers = len(self.hidden_dim)
        
        # print(f"Hidden dim: {self.hidden_dim}")
        
        self.linear = torch.nn.ModuleList()
        for idx, numb in enumerate(self.hidden_dim):
            if idx == 0:
                self.linear.append(nn.Linear(input_dim, numb))
            else:
                self.linear.append(nn.Linear(self.hidden_dim[idx-1], numb))
                
        self.batchnorm = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()   
        
        for idx in range(self.numb_layers-1):
            if batchnorm:
                self.batchnorm.append(nn.BatchNorm1d(num_features=self.hidden_dim[idx]))
            else:
                self.batchnorm.append(nn.Identity())              
            if dropout is not None:
                self.dropout.append(nn.Dropout(dropout))
            else:
                self.dropout.append(nn.Identity())
                
        if perform_at_end:
            if batchnorm:
                self.batchnorm.append(nn.BatchNorm1d(num_features=output_dim))
            else:
                self.batchnorm.append(nn.Identity())     
            if dropout is not None:
                self.dropout.append(nn.Dropout(dropout))
            else:
                self.dropout.append(nn.Identity())
        # print(f"Summary MLP: No Linear: {len(self.linear)} --- No BN: {len(self.batchnorm)} --- No DO: {len(self.dropout)}")
              
    def forward(self, x):
        # x should has format of [1, dim]
        # print(f'MLP Network input: {x.shape} --- numb_layer: {self.numb_layers}')
        for i in range(self.numb_layers-1):
            if self.use_residual:
                x = self.linear[i](x) + x
            else:
                x = self.linear[i](x)
            # print(x.shape)
            x = self.batchnorm[i](x)
            # print(x.shape)
            x = self.activate(x)
            # print(x.shape)
            x = self.dropout[i](x)
            # print(x.shape)
            # print('--------')
        if self.perform_at_end:
            # print(f"Perfrom at end --- {x.shape}")
            if self.use_residual:
                x = self.linear[self.numb_layers-1](x) + x
            else:
                x = self.linear[self.numb_layers-1](x)
            # print(x.shape)
            x = self.batchnorm[self.numb_layers-1](x)
            # print(x.shape)
            x = self.activate(x)
            # print(x.shape)
            x = self.dropout[self.numb_layers-1](x)
            # print(x.shape)
        else:
            x = self.linear[self.numb_layers-1](x)
            
        return x

# Attention Mechanism to embed entire graph into 1 vector
class ATT_Layer(nn.Module) :
    def __init__(self, unit_dim=300, init=True):
        super(ATT_Layer, self).__init__()
        self.unit_dim = unit_dim
        self.theta = nn.Parameter(torch.rand(unit_dim, unit_dim))
        self.activation = nn.ReLU(inplace=True)
        self.out_activation = nn.Sigmoid()
        if init:
            nn.init.kaiming_normal_(self.theta, mode='fan_out', nonlinearity='relu')
    def forward(self, embed_vec):
        # embed_vec [n_obj x 300]
        mean_unit = torch.mean(embed_vec, 0) # unit_dim shape
        common = self.activation(torch.matmul(self.theta, mean_unit)) # unit_dim shape
        sigmoid = self.out_activation(torch.matmul(embed_vec, common)) # n_unit shape
        new_embed = torch.mean(torch.mul(embed_vec, sigmoid.view(-1, 1)), 0) # mean of (n_unit x unit_dim shape)
        return new_embed
    
class Visual_Feature(nn.Module):
    # Just a FC convert feature extracted from EfficientNet to specific dim
    def __init__(self, input_dim, output_dim=2048, batchnorm=False, activate_fn='tanh'):
        # structure only b0 or b4
        super(Visual_Feature, self).__init__()
        self.activate_fn = activate_fn
        if batchnorm:
            self.bn = nn.BatchNorm1d(num_features=output_dim)
        else:
            self.bn = None
        self.fc = nn.Linear(input_dim, output_dim)
        if self.activate_fn.lower() == 'relu':
            self.activate = nn.ReLU()
        elif self.activate_fn.lower() == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.2)
        elif self.activate_fn.lower() == 'tanh' :
            self.activate = nn.Tanh()
        else:
            self.activate = MemoryEfficientSwish()
            
    def forward(self, inputs):
        #bs = inputs.size(0)
        x = self.fc(inputs)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activate(x)
        return x
    
class Fusion_Layer(nn.Module):
    # Extract feature from an input images by using efficientnet
    def __init__(self, visual_dim, word_dim, output_dim, dropout=None, batchnorm=False, activate_fn='tanh'):
        # structure only b0 or b4
        super(Fusion_Layer, self).__init__()
        self.input_dim = visual_dim + word_dim
        self.output_dim = output_dim
        if batchnorm:
            self.bn = nn.BatchNorm1d(num_features=self.output_dim)
        else:
            self.bn = None
        if dropout is not None:
            self.do = nn.Dropout(dropout)
        else:
            self.do = None    
        self.fc = nn.Linear(self.input_dim, self.output_dim)
        self.activate_fn = activate_fn
        if self.activate_fn.lower() == 'relu':
            self.activate = nn.ReLU()
        elif self.activate_fn.lower() == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.2)
        elif self.activate_fn.lower() == 'tanh' :
            self.activate = nn.Tanh()
        else:
            self.activate = MemoryEfficientSwish()
    def forward(self, x_ft, x_emb):
        # x_ft feature extracted from visual images [N_x, ft_dim]
        # x_emb embedding feature from word2vec/glove [N_x, 300]
        x = torch.cat((x_ft, x_emb), dim=1)
        if self.do is not None:
            x = self.do(x)
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activate(x)
        return x
    
# GCN Layer to embed objects and predicates
class GCN_Layer(nn.Module):
    def __init__(self, input_dim=300, pred_dim=300, hidden_dim=[300], output_dim=300, activate_fn='swish', batchnorm=True, dropout=None, last_layer=False, use_residual=False):
        super(GCN_Layer, self).__init__()
        perform_at_end = not last_layer
        '''
        node_kwargs = {
          'input_dim': input_dim,
          'hidden_dim': hidden_dim,
          'output_dim':output_dim,
          'activate_fn': activate_fn,
          'batchnorm':batchnorm,
          'dropout': dropout,
          'perform_at_end': not last_layer,
        }
        edge_kwargs = {
          'input_dim': 3*input_dim,
          'hidden_dim': hidden_dim,
          'output_dim':output_dim,
          'activate_fn': activate_fn,
          'batchnorm':batchnorm,
          'dropout': dropout,
          'perform_at_end': not last_layer,
        }
        '''
        # self.gin_node = MLP(**node_kwargs)
        # self.gin_edge = MLP(**edge_kwargs)
        self.gcn_node = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,\
                            activate_fn=activate_fn, batchnorm=batchnorm, dropout=dropout,\
                            perform_at_end=perform_at_end, use_residual=use_residual)
        self.gcn_edge = MLP(input_dim=2*input_dim+pred_dim, hidden_dim=hidden_dim, output_dim=output_dim,\
                            activate_fn=activate_fn, batchnorm=batchnorm, dropout=dropout,\
                            perform_at_end=perform_at_end, use_residual=use_residual)
        
    def forward(self, embed_objects, embed_predicates, edges):
        '''
        embed_objects and embed_predicates are embeded by embedding layers in the previous stage
        They have size of [n_object, 300], [n_predicates, 300]
        edges is index matrix [n_predicates, 2]
        adjacency is matrix [n_object, n_object] indicate the edge index connecting between 2 nodes
        '''
        # print(f"GIN_Layer: Update Edge")
        # Break apart indices for subjects and objects; these have shape (n_predicates,)
        if embed_predicates.shape[0] > 0:
            s_idx = edges[:, 0].contiguous()
            o_idx = edges[:, 1].contiguous()

            # Get current vectors for subjects and objects; these have shape (n_predicates, 300)
            cur_s_vecs = embed_objects[s_idx]
            cur_o_vecs = embed_objects[o_idx]

            # Update predicates based on edges
            edge_input = torch.cat((cur_s_vecs, embed_predicates, cur_o_vecs), dim=1)
            new_predicates = self.gcn_edge(edge_input)
        else:
            new_predicates = embed_predicates
        # print(f"GIN_Layer: Done Update Edge")
        # Update nodes
        new_objects = self.gcn_node(embed_objects)
        # print(f"GIN_Layer: Done Update Node")
        return new_objects, new_predicates

# Graph embedding Network    
class GCN_Network(nn.Module):
    def __init__(self, gcn_input_dim=300, gcn_pred_dim=300, gcn_output_dim=300, gcn_hidden_dim=[300], numb_gcn_layers=5, batchnorm=True, dropout=None, activate_fn='swish', use_residual=False):
        super(GCN_Network, self).__init__()
        self.numb_gcn_layers = numb_gcn_layers
        self.use_residual = use_residual
        self.use_residual = False # Cannot run in True this time
        '''
        layer_kwargs = {
          'input_dim': gin_input_dim,
          'hidden_dim': gin_hidden_dim,
          'output_dim': gin_output_dim,
          'activate_fn': activate_fn, # swish, relu, or leakyrelu
          'batchnorm': batchnorm,
          'dropout': dropout,
          # 'last_layer': False,
        }
        '''
        self.gcn_layers = torch.nn.ModuleList()
        if self.numb_gcn_layers == 1:
            self.gcn_layers.append(GCN_Layer(last_layer=False, input_dim=gcn_input_dim, pred_dim=gcn_pred_dim,\
                                             hidden_dim=gcn_hidden_dim, \
                                             output_dim=gcn_output_dim, activate_fn=activate_fn, batchnorm=batchnorm, \
                                             dropout=dropout, use_residual=self.use_residual))
        else:
            self.gcn_layers.append(GCN_Layer(last_layer=False, input_dim=gcn_input_dim, pred_dim=gcn_pred_dim, \
                                             hidden_dim=gcn_hidden_dim, \
                                             output_dim=gcn_output_dim, activate_fn=activate_fn, batchnorm=batchnorm, \
                                             dropout=dropout, use_residual=self.use_residual))
            
            for i in range(self.numb_gcn_layers - 2):
                 # self.gin_layers.append(GIN_Layer(last_layer=False, **layer_kwargs))
                self.gcn_layers.append(GCN_Layer(last_layer=False, input_dim=gcn_output_dim, pred_dim=gcn_output_dim, \
                                                 hidden_dim=gcn_hidden_dim, \
                                                 output_dim=gcn_output_dim, activate_fn=activate_fn, batchnorm=batchnorm, \
                                                 dropout=dropout, use_residual=self.use_residual))

            self.gcn_layers.append(GCN_Layer(last_layer=False, input_dim=gcn_output_dim, pred_dim=gcn_output_dim,
                                             hidden_dim=gcn_hidden_dim, \
                                             output_dim=gcn_output_dim, activate_fn=activate_fn, batchnorm=batchnorm, \
                                             dropout=dropout, use_residual=self.use_residual))
        
    def forward(self, embed_objects, embed_predicates, edges):
        '''
        embed_objects and embed_predicates are embeded by embedding layers in the previous stage
        They have size of [n_object, 300], [n_predicates, 300]
        edges is index matrix [n_predicates, 2]
        adjacency is matrix [n_object, n_object] indicate the edge index connecting between 2 nodes
        numb_objects is the list indicating number of objects in each graph (since this is for batch data) --> len(numb_objects) = batch_size
        numb_predicates is the list indicating number of predicates in each graph (since this is for batch data) --> len(numb_predicates) = batch_size
        RETURN:
        Graph embedded vectors [batchsize, graph_embed_dim]
        Embed Objects and Embed Predicates after GIN layers [total_n_object (or total_n_predicates), 300]
        '''

        graph_emb_o = []
        graph_emb_p = []
        list_emb_o = [embed_objects]
        list_emb_p = [embed_predicates]
        for idx, gcn_layer in enumerate(self.gcn_layers):
            #print(f"Processing GIN_LAYERS No {idx}")
            emb_o, emb_p = gcn_layer(list_emb_o[idx], list_emb_p[idx], edges)
            if self.use_residual and (idx+1) % 2 == 0:
                emb_o = emb_o + list_emb_o[idx-1]
                emb_p = emb_p + list_emb_p[idx-1]
            list_emb_o.append(emb_o)
            list_emb_p.append(emb_p)
        
        return list_emb_o[self.numb_gcn_layers], list_emb_p[self.numb_gcn_layers]

# MODEL FOR IMAGE BRANCH (EXCEPT THE EMBEDDING PART)    
class ImageModel(nn.Module):
    # receive the visual images, objects, predicates, edges
    # perform word embedding --> extract visual ft --> fusion --> GCN
    def __init__(self, word_unit_dim=300, gcn_output_dim=300, gcn_hidden_dim=[300], numb_gcn_layers=5, batchnorm=True, dropout=None, activate_fn='swish', visualft_structure='b0', visualft_feature_dim=1024, fusion_output_dim=1024, numb_total_obj=150, numb_total_pred=50, init_weight_obj=None, init_weight_pred=None, include_pred_ft=True):
        
        super(ImageModel, self).__init__()
        self.include_pred_ft = include_pred_ft
        # Embed by Word2Vec/Glove
        self.embed_obj_model = WordEmbedding(numb_words=numb_total_obj, embed_dim=word_unit_dim, 
                                                    init_weight=init_weight_obj, sparse=False)
        self.embed_pred_model = WordEmbedding(numb_words=numb_total_pred, embed_dim=word_unit_dim, 
                                                     init_weight=init_weight_pred, sparse=False)
        # Extract images features by EfficientNet
        if visualft_structure == 'b0':
            effnet_dim = 1280
        if visualft_structure == 'b4':
            effnet_dim = 1792
        if visualft_structure == 'b5':
            effnet_dim = 2048
        
        if visualft_feature_dim == effnet_dim:
            self.visual_extract_obj_model = None
        else:
            self.visual_extract_obj_model = Visual_Feature(input_dim=effnet_dim, 
                                                           output_dim=visualft_feature_dim,
                                                           batchnorm=batchnorm, 
                                                           activate_fn=activate_fn)
        #self.visual_extract_pred_model = Visual_Feature(input_dim=effnet_dim, 
        #                                               output_dim=visualft_feature_dim,
        #                                               batchnorm=batchnorm, 
        #                                               activate_fn=activate_fn)
        # Fusion with word embedding
        self.fusion_obj_model = Fusion_Layer(visual_dim=visualft_feature_dim, word_dim=word_unit_dim, 
                                             output_dim=fusion_output_dim, 
                                             dropout=dropout, batchnorm=batchnorm, 
                                             activate_fn=activate_fn)
        if self.include_pred_ft:
            self.fusion_pred_model = Fusion_Layer(visual_dim=visualft_feature_dim, word_dim=word_unit_dim, 
                                                 output_dim=fusion_output_dim, 
                                                 dropout=dropout, batchnorm=batchnorm, 
                                                 activate_fn=activate_fn)
            pred_dim = fusion_output_dim
        else:
            self.fusion_pred_model = None
            pred_dim = word_unit_dim
            
        # GraphNet
        self.gcn_model = GCN_Network(gcn_input_dim=fusion_output_dim, gcn_pred_dim=pred_dim, 
                                     gcn_output_dim=gcn_output_dim, gcn_hidden_dim=gcn_hidden_dim, 
                                     numb_gcn_layers=numb_gcn_layers, batchnorm=batchnorm, dropout=dropout, 
                                     activate_fn=activate_fn, use_residual=False)
        
    def forward(self, images_objects, images_predicates, list_objects, list_predicates, edges):
        # images_objects [N_obj, 3, 224, 224] images tensor
        # images_predicates [N_pred, 3, 224, 224] images tensor
        # list_objects [N_obj,] tensor
        # list_predicates [N_pred,] tensor
        # edges [N_pred, 2] tensor
        eb_objs = self.embed_obj_model(list_objects)        
        eb_pred = self.embed_pred_model(list_predicates) # embedding
        if self.visual_extract_obj_model is not None:
            images_obj_ft = self.visual_extract_obj_model(images_objects)
            if images_predicates is not None:
                images_pred_ft = self.visual_extract_obj_model(images_predicates)
        else:
            images_obj_ft = images_objects
            if images_predicates is not None:
                images_pred_ft = images_predicates
        fusion_objs= self.fusion_obj_model(images_obj_ft, eb_objs)
        if images_predicates is not None and self.fusion_pred_model is not None:
            fusion_pred= self.fusion_pred_model(images_pred_ft, eb_pred)
        else:
            fusion_pred = eb_pred
        objects, predicates = self.gcn_model(fusion_objs, fusion_pred, edges)
        return objects, predicates
    
    
# WORD EMBEDDING
class WordEmbedding(nn.Module):
    def __init__(self, numb_words, embed_dim=300, init_weight=None, sparse=False):
        '''
        numb_words: int total number of words in the dictionary
        embed_dim: int size of embedding of a word
        init_weight: torch.FloatTensor the initialized weight for the module
        spares: boolean if True, gradient w.r.t. weight matrix will be a sparse tensor
        '''
        super(WordEmbedding, self).__init__()
        self.numb_words = numb_words
        self.embed_dim = embed_dim
        self.init_weight = init_weight
        self.model = nn.Embedding(num_embeddings=self.numb_words, embedding_dim=self.embed_dim, sparse=sparse)
        if self.init_weight is not None:
            print('Initilised with given init_weight')
            self.model.weight.data.copy_(init_weight);
        else:
            print('Randomly Initilised')
            
    def forward(self, x):
        return self.model(x)

# Sentence Model (RNN)
class SentenceModel(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512, numb_layers=2, dropout=0.5, bidirectional=False, structure='GRU'):
        '''
        input_dim: int dim of input of sequence model (300)
        hidden_dim: int dim of hidden state of the sequence model (hidden state dim) (512)
        numb_layers: int number of sequence model (2)
        dropout: float dropout percent
        bidirectional: boolean apply bidirectional or not
        '''
        super(SentenceModel, self).__init__()
        if structure == 'GRU':
            model = nn.GRU
        else:
            model = nn.LSTM
        self.model = model(input_size=input_dim, hidden_size=hidden_dim, num_layers=numb_layers, 
                           batch_first=True, dropout=dropout, bidirectional=bidirectional).to(device)
            
        self.numb_directions = 2 if bidirectional else 1
        self.numb_layers = numb_layers
        self.hidden_state = None
        self.hidden_dim = hidden_dim
        self.structure = structure
        
    def init_hidden(self, batch_size):
        if self.structure == 'GRU':
            return torch.zeros(self.numb_layers * self.numb_directions, batch_size, self.hidden_dim).to(self.device)
        else:
            return (torch.zeros(self.numb_layers * self.numb_directions, batch_size, self.hidden_dim).to(self.device),
                    torch.zeros(self.numb_layers * self.numb_directions, batch_size, self.hidden_dim).to(self.device))
        
    def forward(self, x, len_original_x):
        '''
        x is padded and embedded
        len_original_x is the len of un_padded of x
        '''
        batch_size, max_seq_len, input_dim = x.shape
        packed = pack_padded_sequence(x, len_original_x, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed) # self.hidden
        out_unpacked, lens_out = pad_packed_sequence(output, batch_first=True)
        if self.numb_directions == 2:
            out_unpacked_combine = (out_unpacked[:,:,:int(out_unpacked.shape[-1]/2)] + out_unpacked[:,:,int(out_unpacked.shape[-1]/2):])/2
        else:
            out_unpacked_combine = out_unpacked
        '''
        # Extract last hidden state
        if self.structure == 'GRU':
            final_state = hidden.view(self.numb_layers, self.numb_directions, batch_size, self.hidden_dim)[-1]
        elif:
            final_state = hidden[0].view(self.numb_layers, self.numb_directions, batch_size, self.hidden_dim)[-1]
        # Handle directions
        final_hidden_state = None
        if self.num_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        elif self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            # final_hidden_state = h_1 + h_2               # Add both states (requires changes to the input size of first linear layer + attention layer)
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states
        '''
        
        return out_unpacked_combine # batch, max seq len, hidden_size

# Relations Model (for caption) (RNN)    
class RelsModel(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512, numb_layers=2, dropout=0.5, bidirectional=False, structure='GRU'):
        super(RelsModel, self).__init__()
        if structure == 'GRU':
            model = nn.GRU
        else:
            model = nn.LSTM
        self.model = model(input_size=input_dim, hidden_size=hidden_dim, num_layers=numb_layers, 
                           batch_first=True, dropout=dropout, bidirectional=bidirectional).to(device)
            
        self.numb_directions = 2 if bidirectional else 1
        self.numb_layers = numb_layers
        self.hidden_state = None
        self.hidden_dim = hidden_dim
        self.structure = structure
        
    def forward(self, x, len_original_x):
        
        batch_size, max_seq_len, input_dim = x.shape
        packed = pack_padded_sequence(x, len_original_x, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed) # self.hidden
        
        out_unpacked, lens_out = pad_packed_sequence(output, batch_first=True) # state of each subject, pred, object
        if self.numb_directions == 2:
            out_unpacked_combine = (out_unpacked[:,:,:int(out_unpacked.shape[-1]/2)] + out_unpacked[:,:,int(out_unpacked.shape[-1]/2):])/2
        else:
            out_unpacked_combine = out_unpacked # batch, max seq len, hidden_size
        
        # Extract last hidden state
        if self.structure == 'GRU':
            final_state = hidden.view(self.numb_layers, self.numb_directions, batch_size, self.hidden_dim)[-1]
        else:
            final_state = hidden[0].view(self.numb_layers, self.numb_directions, batch_size, self.hidden_dim)[-1]
        
        # Handle directions
        if self.numb_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        else:
            h_1, h_2 = final_state[0], final_state[1]
            final_hidden_state = (h_1 + h_2)/2               # Add both states (requires changes to the input size of first linear layer + attention layer)
            #final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states
            
        return final_hidden_state, out_unpacked_combine
    
class GraphEmb(nn.Module):
    def __init__(self, node_dim=1024, edge_dim=1024, fusion_dim=None, activate_fn='swish', batchnorm=True, dropout=None):
        '''
        Embed a graph with nodes (node_dim) and edges (edge_dim) into a vector
        '''
        super(GraphEmb, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        if fusion_dim is None:
            self.fusion = None
        else:
            self.fusion = MLP(input_dim=self.node_dim+self.edge_dim, hidden_dim=[], output_dim=fusion_dim,\
                              activate_fn=activate_fn, batchnorm=batchnorm, dropout=dropout,\
                              perform_at_end=False, use_residual=False)
        self.att_layers_obj = ATT_Layer(unit_dim=node_dim, init=True)
        self.att_layers_pred = ATT_Layer(unit_dim=edge_dim, init=True)
        
    def forward(self, eb_nodes, eb_edges, numb_nodes, numb_edges):
        '''
        eb_nodes [n_node, node_dims]
        eb_edges [n_edge, edge_dims]
        numb_nodes [list batch] number of nodes in each graph
        numb_edges [list batch] number of edges in each graph
        '''
        count_o = 0 # object = node
        count_p = 0 # pred = edges
        geb = torch.zeros(len(numb_nodes), self.node_dim+self.edge_dim).to(device)
        for idx_batch in range(len(numb_nodes)):
            numb_obj = numb_nodes[idx_batch]
            if numb_obj > 0:
                cur_e_o = eb_nodes[count_o:(count_o+numb_obj)]
            else:
                cur_e_o = torch.zeros((1,self.node_dim)).to(device)
            count_o += numb_obj
            graph_emb_o_l = self.att_layers_obj(cur_e_o).view(1,-1) # convert to [1, dim]

            numb_pred = numb_edges[idx_batch]
            if numb_pred > 0:
                cur_e_p = eb_edges[count_p:(count_p+numb_pred)]
            else:
                cur_e_p = torch.zeros((1,self.edge_dim)).to(device)
            count_p += numb_pred
            graph_emb_p_l = self.att_layers_pred(cur_e_p).view(1,-1) # convert to [1, dim]
            geb[idx_batch] = torch.cat((graph_emb_o_l, graph_emb_p_l), dim=1)
        if self.fusion is not None:
            geb = self.fusion(geb)
        return geb
    
# Discimator between 2 vectors (or 2 embeded graphs)
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, activate_fn='swish', batchnorm=True, dropout=None):
        # hidden dim should be a list, or empty list
        
        super(Discriminator, self).__init__()
        output_dim = 1
        
        if activate_fn.lower() == 'relu':
            self.activate = nn.ReLU()
        elif activate_fn.lower() == 'leakyrelu':
            self.activate = nn.LeakyReLU()
        elif activate_fn.lower() == 'tanh' :
            self.activate = nn.Tanh()
        elif activate_fn.lower() == 'ptanh':
            self.activate = PTanh(0.25)
        else:
            self.activate = MemoryEfficientSwish()
         
        self.hidden_dim = hidden_dim + [output_dim]
        self.numb_layers = len(self.hidden_dim)
        
        self.linear = torch.nn.ModuleList()
        for idx, numb in enumerate(self.hidden_dim):
            if idx == 0:
                self.linear.append(nn.Linear(input_dim, numb))
            else:
                self.linear.append(nn.Linear(self.hidden_dim[idx-1], numb))
                
        self.batchnorm = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()   
        
        for idx in range(self.numb_layers-1):
            if batchnorm:
                self.batchnorm.append(nn.BatchNorm1d(num_features=self.hidden_dim[idx]))
            else:
                self.batchnorm.append(nn.Identity())              
            if dropout is not None:
                self.dropout.append(nn.Dropout(dropout))
            else:
                self.dropout.append(nn.Identity())
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, feat1, feat2):
        dist = torch.abs(feat1-feat2)
        mul = torch.mul(feat1, feat2)
        ft = torch.cat([feat1, feat2, dist, mul], dim=1)
        
        for i in range(self.numb_layers-1):
            ft = self.linear[i](ft)
            ft = self.batchnorm[i](ft)
            ft = self.activate(ft)
            ft = self.dropout[i](ft)
            
        ft = self.linear[self.numb_layers-1](ft)
        ft = self.sigmoid(ft)
            
        return ft