import os, json, time, random
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler
from datetime import datetime
import pkbar
from a2radnet.models.factory import build_model
from a2radnet.runtime import prepare_training_loaders
from .dataset import RADIal
from .encoder import ra_encoder
from .dataloader import CreateDataLoaders
from .losses import pixor_loss
from .evaluation import run_evaluation
is_slurm = "SLURM_JOB_ID" in os.environ
def main(config, resume):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    curr_date = datetime.now()
    exp_name = config['name'] + '_' + f"{time.strftime('%Y%m%d_%H%M%S')}"
    print(exp_name)
    output_folder = config['output']['dir']
    os.mkdir(os.path.join(output_folder, exp_name))
    os.mkdir(os.path.join(output_folder, exp_name, 'model'))
    with open(os.path.join(output_folder, exp_name, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(os.path.join(output_folder, exp_name))
    label_map = {'detection': [128, 112], 'segmentation': [256, 224]}
    if 'label_map' in config:
        label_map = config['label_map']
    enc = ra_encoder(geometry=config['dataset']['geometry'], statistics=config['dataset']['statistics'], regression_layer=2, label_map=label_map)
    net = build_model(config)
    if hasattr(net, 'vit'):
        print('patch_size:', net.vit.patch_embed.patch_size)
        print('merge_sizes:', net.vit.merge_sizes)
        print('stage_periods:', net.vit.stage_periods)
        for (i, layer) in enumerate(net.vit.layers):
            print(i, type(layer))
            if hasattr(layer, 'merge_size'):
                print('  merge_size:', layer.merge_size)
                print('  window_size:', layer.window_size)
            if hasattr(layer, 'blocks'):
                for (j, blk) in enumerate(layer.blocks):
                    if hasattr(blk, 'period'):
                        print('  block', j, 'period', blk.period, 'range_window_size', blk.range_window_size, 'range_shift_size', blk.range_shift_size)
    if hasattr(net, 'frontend'):
        print('Frontend type:', type(net.frontend))
        if net.frontend is not None and hasattr(net.frontend, 'embed_dim'):
            print('  embed_dim:', net.frontend.embed_dim)
            print('  doppler_reducer:', net.frontend.reducer)
            print('  angle_estimator:', net.frontend.angle_estimator)
            print('  num_latents:', net.frontend.num_latents)
            print('  out_channels:', net.frontend.out_channels)
    dataset = RADIal(root_dir=config['dataset']['root_dir'], statistics=config['dataset']['statistics'], encoder=enc.encode, difficult=True, perform_FFT=config['data_mode'], cfg=config)
    (train_loader, val_loader, test_loader) = CreateDataLoaders(dataset, config['dataloader'], config['seed'])
    train_loader, val_loader, test_loader = prepare_training_loaders((train_loader, val_loader, test_loader), config, os.path.join(output_folder, exp_name))
    t_params = sum((p.numel() for p in net.parameters()))
    print('Network Parameters: ', t_params)
    net.to('cuda')
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    num_epochs = int(config['num_epochs'])
    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')
    startEpoch = 0
    global_step = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'F1': [], 'mIoU': []}
    best_mAP = 0
    freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')
    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch'] + 1
        history = dict['history']
        global_step = dict['global_step']
        print('       ... Start at epoch:', startEpoch)
    for epoch in range(startEpoch, num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False, verbose=0 if is_slurm else 1)
        net.train()
        running_loss = 0.0
        for (i, data) in enumerate(train_loader):
            batch_size = data[0].size(0)
            inputs = data[0].to('cuda').float()
            label_map = data[1].to('cuda').float()
            if config['model']['SegmentationHead'] == True:
                seg_map_label = data[2].to('cuda').double()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = net(inputs)
            (classif_loss, reg_loss) = pixor_loss(outputs['Detection'], label_map, config['losses'])
            prediction = outputs['Segmentation'].contiguous().flatten()
            label = seg_map_label.contiguous().flatten()
            loss_seg = freespace_loss(prediction, label)
            loss_seg *= batch_size
            classif_loss *= config['losses']['weight'][0]
            reg_loss *= config['losses']['weight'][1]
            loss_seg *= config['losses']['weight'][2]
            loss = classif_loss + reg_loss + loss_seg
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
            writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
            writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)
            running_loss += loss.item() * batch_size
            kbar.update(i, values=[('loss', loss.item()), ('class', classif_loss.item()), ('reg', reg_loss.item()), ('freeSpace', loss_seg.item())])
            global_step += 1
        scheduler.step()
        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])
        eval = run_evaluation(net, val_loader, enc, check_perf=epoch >= 8, detection_loss=pixor_loss, segmentation_loss=freespace_loss, losses_params=config['losses'], config=config)
        F1 = 2 * eval['mAP'] * eval['mAR'] / (eval['mAP'] + eval['mAR']) if eval['mAP'] + eval['mAR'] > 0 else 0
        history['val_loss'].append(eval['loss'])
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])
        history['F1'].append(F1)
        history['mIoU'].append(eval['mIoU'])
        kbar.add(1, values=[('val_loss', eval['loss']), ('mAP', eval['mAP']), ('mAR', eval['mAR']), ('F1', F1), ('mIoU', eval['mIoU'])])
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Loss/test', eval['loss'], global_step)
        writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
        writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)
        writer.add_scalar('Metrics/F1', F1, global_step)
        writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)
        name_output_file = config['name'] + '_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'], eval['mAP'], eval['mAR'], eval['mIoU'])
        filename = os.path.join(output_folder, exp_name, 'model', name_output_file)
        checkpoint = {}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step
        torch.save(checkpoint, filename)
        print('')
    writer.close()
    return os.path.join(output_folder, exp_name)
