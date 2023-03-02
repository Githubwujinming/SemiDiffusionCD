from collections import OrderedDict
import logging
import os
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from misc.metric_tools import ConfuseMatrixMeter
from misc.torchutils import get_scheduler

from model import networks
from .base_model import BaseModel
from core.pertubs import *
from core.losses import *
import torch.nn.functional as F
from core.losses import consistency_weight

logger = logging.getLogger('base')
class Consistency_Diffusion_CD(BaseModel):
    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.enable_amp = True if torch.cuda.is_available() and opt['enable_amp'] else False
        opt['enable_amp'] = self.enable_amp
        # define network and load pretrained models
        self.netCD = self.set_device(networks.define_SemiCD(opt))# NOTE: cd head 要放在前面，因为feat_scales会被倒转
        self.netG = self.set_device(networks.define_G(opt))
        
        self.schedule_phase = None
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        # set loss and load resume state
        self.loss_type = opt['model_cd']['loss_type']
        if self.loss_type == 'ce':
            self.sup_loss_func =nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplementedError()
        # Supervised and unsupervised losses
        if opt['semi_train']['un_loss']  == "KL":
            self.unsuper_loss_func = softmax_kl_loss
        elif opt['semi_train']['un_loss'] == "MSE":
            self.unsuper_loss_func = softmax_mse_loss
        elif opt['semi_train']['un_loss'] == "JS":
            self.unsuper_loss_func = softmax_js_loss
        else:
            raise ValueError(f"Invalid supervised loss {opt['semi_train']['un_loss']}")
     
        self.steps = opt['model_cd']['t']
        if 'train' in self.opt['phase'] :
            self.netG.train()
            self.netCD.train()
            self.training = True
            # find the parameters to optimize
            optim_cd_params = list(self.netCD.parameters())
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_diffusion_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_diffusion_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_diffusion_params = list(self.netG.parameters())
            

            if opt['train']["optimizer"]["type"] == "adam":
                self.optCD = torch.optim.Adam(
                    optim_cd_params, lr=opt['train']["optimizer"]["lr"])
                self.optG = torch.optim.Adam(
                    optim_diffusion_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optCD = torch.optim.AdamW(
                    optim_cd_params, lr=opt['train']["optimizer"]["lr"])
                self.optG = torch.optim.AdamW(
                    optim_diffusion_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))
            
            self.log_dict = OrderedDict()
            
            #Define learning rate sheduler
            self.exp_lr_scheduler_netCD = get_scheduler(optimizer=self.optCD, args=opt['train'])
        else:
            self.training = False
            self.netCD.eval()
            self.log_dict = OrderedDict()
        self.feat_type = opt['model_cd']['feat_type']
        self.load_network()
        
        # semi cd settings
        self.sup_running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        self.unsup_running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        self.len_train_dataloader = opt["len_train_dataloader"]
        self.len_val_dataloader = opt["len_val_dataloader"]
        
        # self.print_network()
        self.unsup_loss_w   = consistency_weight(final_w=opt['semi_train']['unsupervised_w'], iters_per_epoch=opt['len_semi_train_dataloader'],
                                        rampup_ends=int(opt['semi_train']['ramp_up'] * opt['train']['n_epoch']))
        self.sup_loss_w     = opt['semi_train']['supervised_w']
        self.softmax_temp   = opt['semi_train']['softmax_temp']
        
        # confidence masking (sup mat)
        self.confidence_th      = opt['semi_train']['confidence_th']
        self.confidence_masking = opt['semi_train']['confidence_masking']
        
        # add others pertubs
        
        self.pertubs = self._make_perturbs(opt)
        
        # 在训练最开始之前实例化一个GradScaler对象
        
        self.scaler = amp.GradScaler(enabled=self.enable_amp)
    
    def eval(self):
        self.training = False
        
    def train(self):
        self.training = True
        
    def _make_perturbs(self, opt):
        
        vat     = [VAT(xi=opt['semi_train']['xi'], eps=opt['semi_train']['eps']) for _ in range(opt['semi_train']['vat'])]
        drop    = [DropOut(drop_rate=opt['semi_train']['drop_rate'], spatial_dropout=opt['semi_train']['spatial'])
                                    for _ in range(opt['semi_train']['drop'])]
        cut     = [CutOut(erase=opt['semi_train']['erase']) for _ in range(opt['semi_train']['cutout'])]
        context_m = [ContextMasking() for _ in range(opt['semi_train']['context_masking'])]
        object_masking  = [ObjectMasking() for _ in range(opt['semi_train']['object_masking'])]
        feature_drop    = [FeatureDrop() for _ in range(opt['semi_train']['feature_drop'])]
        feature_noise   = [FeatureNoise(uniform_range=opt['semi_train']['uniform_range'])
                                    for _ in range(opt['semi_train']['feature_noise'])]

        # return nn.ModuleList([*cut,
        #                         *context_m, *object_masking, *feature_drop])
        return nn.ModuleList([*vat, *drop, *cut,
                                *context_m, *object_masking, *feature_drop, *feature_noise])

    # Feeding all data to the CD model
    def istraining(self):
        return self.training
    def feed_data(self, data):
        if self.training:
            self.sup_data, self.unsup_data = data
            self.sup_data = self.set_device(self.sup_data)
            self.unsup_data = self.set_device(self.unsup_data)
        else:
            self.sup_data = self.set_device(data)
    
    # Get feature representations for a given image
    def get_feats(self, t):
        self.netG.eval()
        if self.training:
            sup_data, unsup_data = self.sup_data, self.unsup_data
            Al, Bl = sup_data['A'], sup_data['B']
            Aul, Bul = unsup_data['A'], unsup_data['B']
            with torch.no_grad():
                if isinstance(self.netG, nn.DataParallel):
                    fe_Al, fd_Al = self.netG.module.feats(Al, t)
                    fe_Bl, fd_Bl = self.netG.module.feats(Bl, t)
                    fe_Aul, fd_Aul = self.netG.module.feats(Aul, t)
                    fe_Bul, fd_Bul = self.netG.module.feats(Bul, t)
                else:
                    fe_Al, fd_Al = self.netG.feats(Al, t)
                    fe_Bl, fd_Bl = self.netG.feats(Bl, t)
                    fe_Aul, fd_Aul = self.netG.feats(Aul, t)
                    fe_Bul, fd_Bul = self.netG.feats(Bul, t)
            self.netG.train()
            if self.feat_type == 'dec':
                del fe_Al, fe_Bl, fe_Aul, fe_Bul, Al, Bl, Aul, Bul
                return fd_Al, fd_Bl, fd_Aul, fd_Bul
            else:
                del fd_Al, fd_Bl, fd_Aul, fd_Bul, Al, Bl, Aul, Bul
                return fe_Al, fe_Bl, fe_Aul, fe_Bul
        else:
            sup_data = self.sup_data
            Al, Bl = sup_data['A'], sup_data['B']
            with torch.no_grad():
                if isinstance(self.netG, nn.DataParallel):
                    fe_Al, fd_Al = self.netG.module.feats(Al, t)
                    fe_Bl, fd_Bl = self.netG.module.feats(Bl, t)
                else:
                    fe_Al, fd_Al = self.netG.feats(Al, t)
                    fe_Bl, fd_Bl = self.netG.feats(Bl, t)
            self.netG.train()
            if self.feat_type == 'dec':
                del fe_Al, fe_Bl, Al, Bl
                return fd_Al, fd_Bl
            else:
                del fd_Al, fd_Bl, Al, Bl
                return fe_Al, fe_Bl
    
    def forward(self, epoch, curr_iter):
        f_Al=[]
        f_Bl=[]
        f_Aul=[]
        f_Bul=[]
        for t in self.steps:
            f_Al_t,f_Bl_t, f_Aul_t, f_Bul_t = self.get_feats(t=t) #np.random.randint(low=2, high=8)
            f_Al.append(f_Al_t)
            f_Bl.append(f_Bl_t)
            f_Aul.append(f_Aul_t)
            f_Bul.append(f_Bul_t)
        del f_Al_t,f_Bl_t, f_Aul_t, f_Bul_t
        # Feeding features from the diffusion model to the CD model
        tl = self.sup_data['L']
        self.o_l = self.netCD(f_Al, f_Bl)
        sup_loss = self.sup_loss_func(self.o_l, tl.long())*self.sup_loss_w
        
        # Get main prediction
        self.o_ul = self.netCD(f_Aul, f_Bul)
        # Get auxiliary predictions
        o_uls = [self.netCD(f_Aul, f_Bul, perturbation=pertub, o_l = self.o_ul.detach()) for pertub in self.pertubs]
        targets = F.softmax(self.o_ul.detach(), dim=1)
        
         # Compute unsupervised loss
        unsup_loss = sum([self.unsuper_loss_func(inputs=u, targets=targets, \
                        conf_mask=self.confidence_masking, threshold=self.confidence_th, use_softmax=False)
                        for u in o_uls])
        unsup_loss = (unsup_loss / len(o_uls))
        self.log_dict['sup_loss'] = sup_loss
        del o_uls, targets
        # Compute the unsupervised loss
        weight_u    = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
        unsup_loss  = unsup_loss * weight_u
        self.log_dict['unsup_loss'] = unsup_loss
        total_loss  = unsup_loss  + sup_loss 
        self.log_dict['total_loss'] = total_loss
        return total_loss
        

    
    # Optimize the parameters of the CD model
    def optimize_parameters(self, epoch, curr_iter):
        self.optCD.zero_grad()
        with amp.autocast(enabled=self.enable_amp):
            l_cd = self.forward(epoch, curr_iter)
        if self.enable_amp:
            # 1、Scales loss.  先将梯度放大 防止梯度消失
            self.scaler.scale(l_cd).backward()
            # 2、scaler.step()   再把梯度的值unscale回来.
            self.scaler.step(self.optCD)
            
            self.scaler.update()
            torch.cuda.empty_cache()
        else:
            l_cd.backward()
            self.optCD.step()
        

    # Testing on given data
    def test(self):
        self.netCD.eval()
        f_Al=[]
        f_Bl=[]
        with torch.no_grad():
            for t in self.steps:
                f_Al_t,f_Bl_t = self.get_feats(t=t) #np.random.randint(low=2, high=8)
                f_Al.append(f_Al_t)
                f_Bl.append(f_Bl_t)
            if isinstance(self.netCD, nn.DataParallel):
                self.o_l = self.netCD.module.forward(f_Al, f_Bl)
            else:                    
                self.o_l = self.netCD.forward(f_Al, f_Bl)

            l_cd = self.sup_loss_func(self.o_l, self.sup_data["L"].long())
            self.log_dict['sup_loss'] = l_cd.item()
        self.netCD.train()

    # Get current log
    def get_current_log(self):
        return self.log_dict

    # Get current visuals
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['o_l'] = torch.argmax(self.o_l, dim=1, keepdim=False)
        out_dict['gt_cm'] = self.sup_data['L']
        if self.training:
            out_dict['o_ul'] = torch.argmax(self.o_ul, dim=1, keepdim=False)
            out_dict['gt_u_cm'] = self.unsup_data['L']
        return out_dict

    # Printing the CD network
    def print_network(self):
        s, n = self.get_network_description(self.netCD)
        if isinstance(self.netCD, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netCD.__class__.__name__,
                                             self.netCD.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netCD.__class__.__name__)

        logger.info(
            'Change Detection Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # Saving the network parameters
    def save_network(self, epoch, is_best_model = False):
        cd_gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'semicd_model_E{}_gen.pth'.format(epoch))
        cd_opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'semicd_model_E{}_opt.pth'.format(epoch))
        
        if is_best_model:
            best_cd_gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_semicd_model_gen.pth'.format(epoch))
            best_cd_opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_semicd_model_opt.pth'.format(epoch))

        # Save CD model pareamters
        network = self.netCD
        if isinstance(self.netCD, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, cd_gen_path)
        if is_best_model:
            torch.save(state_dict, best_cd_gen_path)


        # Save CD optimizer paramers
        opt_state = {'epoch': epoch,
                     'scheduler': None, 
                     'optimizer': None}
        opt_state['optimizer'] = self.optCD.state_dict()
        torch.save(opt_state, cd_opt_path)
        if is_best_model:
            torch.save(opt_state, best_cd_opt_path)

        # Print info
        logger.info(
            'Saved current CD model in [{:s}] ...'.format(cd_gen_path))
        if is_best_model:
            logger.info(
            'Saved best CD model in [{:s}] ...'.format(best_cd_gen_path))

    # Loading pre-trained CD network
    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))

            network.load_state_dict(torch.load(
                gen_path), strict=False)
        load_path = self.opt['path_cd']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for CD model [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            
            # change detection model
            network = self.netCD
            if isinstance(self.netCD, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=True)
            
            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optCD.load_state_dict(opt['optimizer'])
                # self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']+1
    
    # Functions related to computing performance metrics for CD
    def _update_metric(self):
        """
        update metric
        """
        Gl_pred = self.o_l.detach()
        Gl_pred = torch.argmax(Gl_pred, dim=1)

        sup_current_score = self.sup_running_metric.update_cm(pr=Gl_pred.cpu().numpy(), gt=self.sup_data['L'].detach().cpu().numpy())
        if self.training:
            Gul_pred = self.o_ul.detach()
            Gul_pred = torch.argmax(Gul_pred, dim=1)

            unsup_current_score = self.unsup_running_metric.update_cm(pr=Gul_pred.cpu().numpy(), gt=self.unsup_data['L'].detach().cpu().numpy())
            return sup_current_score, unsup_current_score
        return sup_current_score
    
    # Collecting status of the current running batch
    def _collect_running_batch_states(self):
        
        if self.training:
            self.sup_running_acc, self.unsup_running_acc = self._update_metric()
            self.log_dict['unsup_running_acc'] = self.unsup_running_acc.item()
        else:
            self.sup_running_acc = self._update_metric()
        self.log_dict['sup_running_acc'] = self.sup_running_acc.item()

    # Collect the status of the epoch
    def _collect_epoch_states(self):
        sup_scores = self.sup_running_metric.get_scores()
        self.sup_epoch_acc = sup_scores['mf1']

        self.log_dict['sup_epoch_acc'] = self.sup_epoch_acc.item()

        for k, v in sup_scores.items():
            self.log_dict['sup_'+k] = v
        if self.training:
            unsup_scores = self.unsup_running_metric.get_scores()
            self.unsup_epoch_acc = unsup_scores['mf1']
            self.log_dict['unsup_epoch_acc'] = self.unsup_epoch_acc.item()
            for k, v in unsup_scores.items():
                self.log_dict['unsup_'+k] = v
                #message += '%s: %.5f ' % (k, v)

    # Rest all the performance metrics
    def _clear_cache(self):
        self.sup_running_metric.clear()
        if self.training:
            self.unsup_running_metric.clear()

    # Finctions related to learning rate sheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netCD.step()
        
    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

        
