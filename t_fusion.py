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
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/fusion_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='test')
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
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            print("Creating test dataloader.")
            test_dataset = FD(split='test',
                              mri_path='dataset/CT-MRI_GEN/test1/MRI',
                              pet_path='dataset/CT-MRI_GEN/test1/CT')
            print("the testing dataset is length:{}".format(test_dataset.length))
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=False,
                pin_memory=True,
                drop_last=False,
            )
            test_loader.n_iter = len(test_loader)

    logger.info('Initial Dataset Finished')

    # Creating Fusion model
    fusion_net = Model.create_model(opt)
    logger.info('Initial Encoder Model Finished')

    logger.info('Begin Model Evaluation (testing).')
    test_result_path = '{}/test/'.format(opt['path']['results'])
    os.makedirs(test_result_path, exist_ok=True)
    logger_test = logging.getLogger('test')  # test logger

    # 初始化总时间
    total_time = 0.0

    for current_step, (test_data, file_names) in enumerate(test_loader):
        start = time.time()
        fusion_net.feed_data(test_data)
        fusion_net.test()
        visuals = fusion_net.get_current_visuals()
        grid_img = visuals['img'].detach()
        # grid_img = ycbcr2rgb(
        #     grid_img, test_data['cb'], test_data['cr'])
        grid_img = Metrics.tensor2img(grid_img)
        Metrics.save_img1(
            grid_img, '{}/{}'.format(test_result_path, file_names[0]))
        end = time.time()  # 记录结束时间
        elapsed_time = end - start  # 计算单次耗时
        total_time += elapsed_time  # 累加总耗时

        logger_test.info(f"Step [{current_step + 1}/{len(test_loader)}], "
                         f"Time: {elapsed_time:.4f}s, "
                         f"File: {file_names[0]}")
    logger.info('End of Testing.')
    # 计算平均耗时
    avg_time = total_time / len(test_loader)
    logger.info(f"End of Testing. Total time: {total_time:.4f}s, "
                f"Average time per image: {avg_time:.4f}s")
