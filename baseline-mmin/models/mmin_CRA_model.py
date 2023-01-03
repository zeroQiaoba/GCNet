import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
from models.networks.autoencoder import ResidualAE
from models.utt_fusion_model import UttFusionModel
from .utils.config import OptConfig


class MMINCRAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--AE_layers', type=str, default='128,64,32', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.dataset = opt.dataset_mode.split('_')[0]
        self.loss_names = ['ce', 'recon']
        self.model_names = ['A', 'AA', 'V', 'VV', 'L', 'LL', 'C', 'AE']
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        
        # acoustic model
        self.netA = FcClassifier(opt.input_dim_a, cls_layers, output_dim=opt.embd_size_a, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netAA = FcClassifier(AE_input_dim, cls_layers, output_dim=opt.input_dim_a, dropout=opt.dropout_rate, use_bn=opt.bn)

        # lexical model
        self.netL = FcClassifier(opt.input_dim_l, cls_layers, output_dim=opt.embd_size_l, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netLL = FcClassifier(AE_input_dim, cls_layers, output_dim=opt.input_dim_l, dropout=opt.dropout_rate, use_bn=opt.bn)

        # visual model
        self.netV = FcClassifier(opt.input_dim_v, cls_layers, output_dim=opt.embd_size_v, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netVV = FcClassifier(AE_input_dim, cls_layers, output_dim=opt.input_dim_v, dropout=opt.dropout_rate, use_bn=opt.bn)

        # AE model
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        # cls_input_size = AE_layers[-1] * opt.n_blocks
        self.netC = FcClassifier(AE_input_dim, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)

        if self.isTrain:
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        acoustic = input['A_feat'].float().to(self.device)
        lexical = input['L_feat'].float().to(self.device)
        visual = input['V_feat'].float().to(self.device)

        self.label = input['label'].to(self.device)
        self.missing_index = input['missing_index'].long().to(self.device)
        # A modality
        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1)
        self.A_miss = acoustic * self.A_miss_index
        self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
        self.A_full = acoustic
        # L modality
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1)
        self.L_miss = lexical * self.L_miss_index
        self.L_reverse = lexical * -1 * (self.L_miss_index - 1)
        self.L_full = lexical
        # V modality
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1)
        self.V_miss = visual * self.V_miss_index
        self.V_reverse = visual * -1 * (self.V_miss_index - 1)
        self.V_full = visual
        


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        ## recon
        self.feat_A_miss, _ = self.netA(self.A_miss)
        self.feat_L_miss, _ = self.netL(self.L_miss)
        self.feat_V_miss, _ = self.netV(self.V_miss)
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)
        self.A_rec, _ = self.netAA(self.recon_fusion)
        self.L_rec, _ = self.netLL(self.recon_fusion)
        self.V_rec, _ = self.netVV(self.recon_fusion)

        ## classifier
        self.hiddens = self.recon_fusion
        self.logits, _ = self.netC(self.recon_fusion)
        self.logits = self.logits.squeeze()
        self.pred = self.logits

       ## calculate cls loss
        if self.dataset in ['cmumosi', 'cmumosei']:                    criterion_ce = torch.nn.MSELoss()
        if self.dataset in ['boxoflies', 'iemocapfour', 'iemocapsix']: criterion_ce = torch.nn.CrossEntropyLoss()
        self.loss_ce = criterion_ce(self.logits, self.label)
        ## calculate recon loss [if miss, the calculate recon loss; if exist, no recon loss]
        recon_loss = torch.nn.MSELoss(reduction='none')
        loss_recon1 = recon_loss(self.A_rec, self.A_full) * -1 * (self.A_miss_index - 1) # 1 (exist), 0 (miss)  [batch, featdim]
        loss_recon2 = recon_loss(self.L_rec, self.L_full) * -1 * (self.L_miss_index - 1) # 1 (exist), 0 (miss)
        loss_recon3 = recon_loss(self.V_rec, self.V_full) * -1 * (self.V_miss_index - 1) # 1 (exist), 0 (miss)
        loss_recon1 = torch.sum(loss_recon1) / self.A_full.shape[1]                   # each dimension delta
        loss_recon2 = torch.sum(loss_recon2) / self.L_full.shape[1]                   # each dimension delta
        loss_recon3 = torch.sum(loss_recon3) / self.V_full.shape[1]                   # each dimension delta
        self.loss_recon = loss_recon1 + loss_recon2 + loss_recon3
        ## merge all loss
        self.loss = self.ce_weight * self.loss_ce + self.mse_weight * self.loss_recon


    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)


    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
