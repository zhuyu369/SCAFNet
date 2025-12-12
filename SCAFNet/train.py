import torch
import models as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
from data.MeDataset import FusionDataset as FD, ycbcr2rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/fusion_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

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
    Logger.setup_logger('train', opt['path']
                        ['log'], 'train', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            print("Creating train dataloader.")
            train_dataset = FD(split='train',
                               mri_path='dataset/CT-MRI_GEN/train1/MRI',
                               pet_path='dataset/CT-MRI_GEN/train1/CT')
            print("the training dataset is length:{}".format(train_dataset.length))
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )
            train_loader.n_iter = len(train_loader)

    logger.info('Initial Dataset Finished')

    # Loading Encoder model
    model = Model.create_model(opt)
    logger.info('Initial Encoder Model Finished')

    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0
    if opt['phase'] == 'train':
        for current_epoch in range(start_epoch, n_epoch):
            train_result_path = '{}/train/{}'.format(opt['path']
                                                     ['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)
            ################
            ### training ###
            ################
            message = 'lr: %0.7f\n \n' % model.optDF.param_groups[0]['lr']
            logger.info(message)
            for current_step, (train_data, _) in enumerate(train_loader):
                # Feeding features
                model.feed_data(train_data)
                model.optimize_parameters()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    model.update_loss()
                    logs = model.get_current_log()
                    message = '[Training FS]. epoch: [%d/%d]. Itter: [%d/%d], ' \
                              'All_loss: %.5f,Intensity_loss: %.5f, Grad_loss: %.5f,SSIM_loss: %.5f' % (current_epoch, n_epoch - 1, current_step, len(train_loader), logs['l_all'],
                               logs['l_in'], logs['l_grad'], logs['l_ssim'])
                    logger.info(message)

            # visuals = model.get_current_visuals()
            # grid_img = visuals['img'].detach()
            # # if train_data['cb'] != None:
            # grid_img = ycbcr2rgb(
            #     grid_img, train_data['cb'], train_data['cr'])

            # grid_img = Metrics.tensor2img(grid_img)
            # Metrics.save_img(grid_img, '{}/img_fused_e{}_b{}.png'.format(train_result_path,
            #                                                              current_epoch,
            #                                                              current_step))
            # else:
            # grid_img = Metrics.tensor2img(grid_img)
            # Metrics.save_img1(grid_img, '{}/img_fused_e{}_b{}.png'.format(train_result_path,
            #                                                                  current_epoch,
            #                                                                  current_step))
            if (current_epoch > 1) & ((current_epoch + 1) % 10 == 0):
                model.save_network(current_epoch)
        model._update_lr_schedulers()
        logger.info('End of fusion training.')
        model.save_network(current_epoch)
