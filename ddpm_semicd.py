from itertools import cycle
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from misc.print_diffuse_feats import print_feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/levir_debug.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        # Training log
        wandb.define_metric('epoch')
        wandb.define_metric('training/train_step')
        wandb.define_metric("training/*", step_metric="train_step")
        # Validation log
        wandb.define_metric('validation/val_step')
        wandb.define_metric("validation/*", step_metric="val_step")
        # Initialization
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None

    # Loading change-detction datasets.
    sup_percent = str(opt['semi_train']['sup_percent'])
    supervised_set = Data.create_cd_dataset(opt['datasets'][sup_percent + '_train_supervised'], sup_percent + '_train_supervised')
    unsupervised_set = Data.create_cd_dataset(opt['datasets'][sup_percent + '_train_unsupervised'],sup_percent + '_train_unsupervised')# NOTE: must add train_unsupervised fields in json
    val_set = Data.create_cd_dataset(opt['datasets']['val'], 'val')
    test_set = Data.create_cd_dataset(opt['datasets']['test'], 'test')
    supervised_loader = Data.create_dataloader(
                supervised_set, opt['datasets'][sup_percent + '_train_supervised'], sup_percent + '_train_supervised')
    unsupervised_loader = Data.create_dataloader(
                unsupervised_set, opt['datasets'][sup_percent + '_train_unsupervised'], sup_percent + '_train_unsupervised')
    val_loader = Data.create_cd_dataloader(val_set, opt['datasets']['val'], 'val')
    test_loader = Data.create_cd_dataloader(test_set, opt['datasets']['test'], 'test')
    opt['len_train_dataloader'] = len(supervised_loader)
    opt['len_semi_train_dataloader'] = len(unsupervised_loader)
    opt['len_val_dataloader'] = len(val_loader)
    logger.info('Initial Dataset Finished')
    

    # Loading diffusion semi-cd model
    semi_cd = Model.create_SemiCD_model(opt)
    logger.info('Initial SemiCD model Finished')

    # Set noise schedule for the diffusion model
    semi_cd.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = semi_cd.begin_epoch
    
    
    if opt['phase'] == 'train':
        semi_cd.train()
        sup_loader = iter(supervised_loader)
        for current_epoch in range(start_epoch, n_epoch):         
            semi_cd._clear_cache()
            train_result_path = '{}/train/{}'.format(opt['path']
                                                 ['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)
            
            ################
            ### training ###
            ################
            message = 'lr: %0.7f\n \n' % semi_cd.optCD.param_groups[0]['lr']
            logger.info(message)
            for current_step, unsup_data in enumerate(unsupervised_loader):
                try:
                    sup_data = next(sup_loader)
                except StopIteration:
                    sup_loader = iter(supervised_loader)
                    sup_data = next(sup_loader)
                # Feeding data to diffusion model and get features
                semi_cd.feed_data((sup_data, unsup_data))
                semi_cd.optimize_parameters(current_epoch, current_step)
                semi_cd._collect_running_batch_states()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    logs = semi_cd.get_current_log()
                    message = '[Training SemiCD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss_sup: %.5f, CD_loss_unsup: %.5f, running_mf1_sup: %.5f, running_mf1_unsup: %.5f\n' %\
                      (current_epoch, n_epoch-1, current_step, len(unsupervised_loader), logs['sup_loss'], logs['unsup_loss'], logs['sup_running_acc'], logs['unsup_running_acc'])
                    logger.info(message)

                    #vissuals
                    visuals = semi_cd.get_current_visuals()

                    img_mode = "grid"
                    if img_mode == "single":
                        # Converting to uint8
                        # sup_data, unsup_data = train_data
                        img_Al   = Metrics.tensor2img(sup_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_Bl   = Metrics.tensor2img(sup_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        gt_cm_l   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                        pred_cm_l = Metrics.tensor2img(visuals['o_l'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                        #save imgs
                        Metrics.save_img(
                                img_Al, '{}/img_Al_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                img_Bl, '{}/img_Bl_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                pred_cm_l, '{}/img_pred_l_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                gt_cm_l, '{}/img_gt_l_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        img_Aul   = Metrics.tensor2img(unsup_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_Bul   = Metrics.tensor2img(unsup_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        gt_cm_ul   = Metrics.tensor2img(visuals['gt_u_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                        pred_cm_ul = Metrics.tensor2img(visuals['o_ul'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                        #save imgs
                        Metrics.save_img(
                                img_Aul, '{}/img_Aul_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                img_Bul, '{}/img_Bul_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                pred_cm_ul, '{}/img_pred_ul_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                gt_cm_ul, '{}/img_gt_ul_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                    else:
                        # grid img
                        visuals['o_l'] = visuals['o_l']*2.0-1.0
                        visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                        # sup_data, unsup_data = train_data
                        grid_img = torch.cat((sup_data['A'], 
                                    sup_data['B'], 
                                    visuals['o_l'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                    visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                    dim = 0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img, '{}/img_A_B_l_pred_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        
                        visuals['o_ul'] = visuals['o_ul']*2.0-1.0
                        visuals['gt_u_cm'] = visuals['gt_u_cm']*2.0-1.0
                        grid_img = torch.cat((unsup_data['A'], 
                                    unsup_data['B'], 
                                    visuals['o_ul'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                    visuals['gt_u_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                    dim = 0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img, '{}/img_A_B_ul_pred_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                
            ### log epoch status ###
            semi_cd._collect_epoch_states()
            logs = semi_cd.get_current_log()
            message = '[Training SemiCD (epoch summary)]: epoch: [%d/%d]. epoch_mF1_sup=%.5f,  epoch_mF1_unsup=%.5f \n' %\
                      (current_epoch, n_epoch-1, logs['sup_epoch_acc'], logs['unsup_epoch_acc'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                tb_logger.add_scalar(k, v, current_step)
            message += '\n'
            logger.info(message)
            
            if wandb_logger:
                    wandb_logger.log_metrics({
                        'training/lr': semi_cd.optCD.param_groups[0]['lr'],
                        'training/sup_mF1': logs['sup_epoch_acc'],
                        'training/sup_mIoU': logs['sup_miou'],
                        'training/sup_OA': logs['sup_acc'],
                        'training/sup_change-F1': logs['sup_F1_1'],
                        'training/sup_no-change-F1': logs['sup_F1_0'],
                        'training/sup_change-IoU': logs['sup_iou_1'],
                        'training/sup_no-change-IoU': logs['sup_iou_0'],
                        
                        'training/unsup_mF1': logs['unsup_epoch_acc'],
                        'training/unsup_mIoU': logs['unsup_miou'],
                        'training/unsup_OA': logs['unsup_acc'],
                        'training/unsup_change-F1': logs['unsup_F1_1'],
                        'training/unsup_no-change-F1': logs['unsup_F1_0'],
                        'training/unsup_change-IoU': logs['unsup_iou_1'],
                        'training/unsup_no-change-IoU': logs['unsup_iou_0'],
                        
                        'training/train_step': current_epoch
                    })

            semi_cd._clear_cache()
            semi_cd._update_lr_schedulers()
            
            ##################
            ### validation ###
            ##################
            semi_cd.eval()
            if current_epoch % opt['train']['val_freq'] == 0:
                val_result_path = '{}/val/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                os.makedirs(val_result_path, exist_ok=True)

                for current_step, val_data in enumerate(val_loader):
                    # Feed data to diffusion model
                    semi_cd.feed_data(val_data)
                    semi_cd.test()
                    semi_cd._collect_running_batch_states()
                    
                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        # message
                        logs        = semi_cd.get_current_log()
                        message     = '[Validation SemiCD]. epoch: [%d/%d]. Itter: [%d/%d], running_mf1: %.5f\n' %\
                                    (current_epoch, n_epoch-1, current_step, len(val_loader), logs['sup_running_acc'])
                        logger.info(message)

                        #vissuals
                        visuals = semi_cd.get_current_visuals()

                        img_mode = "grid"
                        if img_mode == "single":
                            # Converting to uint8
                            img_A   = Metrics.tensor2img(val_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                            img_B   = Metrics.tensor2img(val_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                            gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                            pred_cm = Metrics.tensor2img(visuals['o_l'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                            #save imgs
                            Metrics.save_img(
                                img_A, '{}/img_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                img_B, '{}/img_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                pred_cm, '{}/img_pred_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                gt_cm, '{}/img_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        else:
                            # grid img
                            visuals['o_l'] = visuals['o_l']*2.0-1.0
                            visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                            grid_img = torch.cat((  val_data['A'], 
                                        val_data['B'], 
                                        visuals['o_l'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                        visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                        dim = 0)
                            grid_img = Metrics.tensor2img(grid_img)  # uint8
                            Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))

                semi_cd._collect_epoch_states()
                logs = semi_cd.get_current_log()
                message = '[Validation CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                      (current_epoch, n_epoch-1, logs['sup_epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v) 
                    tb_logger.add_scalar(k, v, current_step)
                message += '\n'
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics({
                        'validation/mF1': logs['sup_epoch_acc'],
                        'validation/mIoU': logs['sup_miou'],
                        'validation/OA': logs['sup_acc'],
                        'validation/change-F1': logs['sup_F1_1'],
                        'validation/no-change-F1': logs['sup_F1_0'],
                        'validation/change-IoU': logs['sup_iou_1'],
                        'validation/no-change-IoU': logs['sup_iou_0'],
                        'validation/val_step': current_epoch               
                    })
                
                if logs['sup_epoch_acc'] > best_mF1:
                    is_best_model = True
                    best_mF1 = logs['sup_epoch_acc']
                    logger.info('[Validation SemiCD] Best model updated. Saving the models (current + best) and training states.')
                else:
                    is_best_model = False
                    logger.info('[Validation SemiCD]Saving the current cd model and training states.')
                logger.info('--- Proceed To The Next Epoch ----\n \n')

                semi_cd.save_network(current_epoch, is_best_model = is_best_model)
                semi_cd._clear_cache()
            semi_cd.train()
            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})
                
        logger.info('End of training.')
    elif opt['phase'] == 'test':
        semi_cd.eval()
        logger.info('Begin Model Evaluation (testing).')
        test_result_path = '{}/test/'.format(opt['path']
                                                 ['results'])
        os.makedirs(test_result_path, exist_ok=True)
        logger_test = logging.getLogger('test')  # test logger
        semi_cd._clear_cache()
        for current_step, test_data in enumerate(test_loader):
            # Feed data to diffusion model
            semi_cd.feed_data(test_data)
            semi_cd.test()
            semi_cd._collect_running_batch_states()

            # Logs
            logs        = semi_cd.get_current_log()
            message     = '[Testing SemiCD]. Itter: [%d/%d], running_mf1: %.5f\n' %\
                                    (current_step, len(test_loader), logs['sup_running_acc'])
            logger_test.info(message)

            # Vissuals
            visuals = semi_cd.get_current_visuals()
            img_mode = 'single'
            if img_mode == 'single':
                # Converting to uint8
                visuals['o_l'] = visuals['o_l']*2.0-1.0
                visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                img_A   = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                img_B   = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                pred_cm = Metrics.tensor2img(visuals['o_l'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                # Save imgs
                Metrics.save_img(
                    img_A, '{}/img_A_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    img_B, '{}/img_B_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    pred_cm, '{}/img_pred_cm{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    gt_cm, '{}/img_gt_cm{}.png'.format(test_result_path, current_step))
            else:
                # grid img
                visuals['o_l'] = visuals['o_l']*2.0-1.0
                visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                grid_img = torch.cat((  test_data['A'], 
                                        test_data['B'], 
                                        visuals['o_l'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                        visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                        dim = 0)
                grid_img = Metrics.tensor2img(grid_img)  # uint8
                Metrics.save_img(
                    grid_img, '{}/img_A_B_pred_gt_{}.png'.format(test_result_path, current_step))

        semi_cd._collect_epoch_states()
        logs = semi_cd.get_current_log()
        message = '[Test CD summary]: Test mF1=%.5f \n' %\
                      (logs['sup_epoch_acc'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            message += '\n'
        logger_test.info(message)

        if wandb_logger:
            wandb_logger.log_metrics({
                'test/mF1': logs['sup_epoch_acc'],
                'test/mIoU': logs['sup_miou'],
                'test/OA': logs['sup_acc'],
                'test/change-F1': logs['sup_F1_1'],
                'test/no-change-F1': logs['sup_F1_0'],
                'test/change-IoU': logs['sup_iou_1'],
                'test/no-change-IoU': logs['sup_iou_0'],
            })

        logger.info('End of testing...')
