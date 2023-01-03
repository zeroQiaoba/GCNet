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


class MMINOldModel(BaseModel):
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
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE', 'mse', 'cycle']
        self.model_names = ['A', 'V', 'L', 'C', 'AE', 'AE_cycle']
        
        # acoustic model
        self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        # lexical model
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        # visual model
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        # AE model
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        self.netAE_cycle = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = AE_layers[-1] * opt.n_blocks
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.miss2_rate = 0.5

        if self.isTrain:
            self.load_pretrained_encoder(opt)
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.cycle_weight = opt.cycle_weight

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def load_pretrained_encoder(self, opt):
        print('Init parameter from {}'.format(opt.pretrained_path))
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')
        pretrained_config = self.load_from_opt_record(pretrained_config_path)
        pretrained_config.isTrain = False                             # teacher model should be in test mode
        pretrained_config.gpu_ids = opt.gpu_ids                       # set gpu to the same
        self.pretrained_encoder = UttFusionModel(pretrained_config)
        self.pretrained_encoder.load_networks_cv(pretrained_path)
        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()
    
    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.'+key, value) for key, value in state_dict.items()])
        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netA.load_state_dict(f(self.pretrained_encoder.netA.state_dict()))
            self.netV.load_state_dict(f(self.pretrained_encoder.netV.state_dict()))
            self.netL.load_state_dict(f(self.pretrained_encoder.netL.state_dict()))
        
    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        acoustic = input['A_feat'].float().to(self.device)
        lexical = input['L_feat'].float().to(self.device)
        visual = input['V_feat'].float().to(self.device)
        if self.isTrain:
            self.label = input['label'].to(self.device)
            batch_size = acoustic.size(0)
            self.missing_index = torch.zeros([batch_size]).long().to(self.device)
            self.miss_index_matrix = torch.zeros([batch_size, 3]).to(self.device)
            self.missing_index = self.missing_index.random_(0, 3)
            self.missing_index = self.miss_index_matrix.scatter_(1, self.missing_index.unsqueeze(1), 1).long()
            self.missing_index[:int((1-self.miss2_rate) * batch_size)] = -1 * \
                            (self.missing_index[:int((1-self.miss2_rate) * batch_size)] - 1) # 前(1-self.miss2_rate)的数据取反, 

            # A modality
            self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
            self.A_miss = acoustic * self.A_miss_index
            self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
            # L modality
            self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
            self.L_miss = lexical * self.L_miss_index
            self.L_reverse = lexical * -1 * (self.L_miss_index - 1)
            # V modality
            self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
            self.V_miss = visual * self.V_miss_index
            self.V_reverse = visual * -1 * (self.V_miss_index - 1)
        else:
            self.A_miss = acoustic
            self.V_miss = visual
            self.L_miss = lexical


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get utt level representattion
        self.feat_A_miss = self.netA(self.A_miss)
        self.feat_L_miss = self.netL(self.L_miss)
        self.feat_V_miss = self.netV(self.V_miss)
        # fusion miss
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)
        # calc reconstruction of teacher's output
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)
        # get fusion outputs for missing modality
        self.logits, _ = self.netC(self.latent)
        self.pred = F.softmax(self.logits, dim=-1)
        # for training 
        if self.isTrain:
            with torch.no_grad():
                self.T_embd_A = self.pretrained_encoder.netA(self.A_reverse)
                self.T_embd_L = self.pretrained_encoder.netL(self.L_reverse)
                self.T_embd_V = self.pretrained_encoder.netV(self.V_reverse)
                self.T_embds = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)
        self.loss_mse = self.mse_weight * self.criterion_mse(self.T_embds, self.recon_fusion)
        self.loss_cycle = self.cycle_weight * self.criterion_mse(self.feat_fusion_miss.detach(), self.recon_cycle)
        loss = self.loss_CE + self.loss_mse + self.loss_cycle
        loss.backward()
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
