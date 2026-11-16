import torch
import numpy as np
from .metrics import GetFullMetrics, Metrics
import pkbar
import pickle
import json
import os
from pathlib import Path
is_slurm = 'SLURM_JOB_ID' in os.environ

def run_evaluation(net, loader, encoder, check_perf=False, detection_loss=None, segmentation_loss=None, losses_params=None, config=None):
    metrics = Metrics()
    metrics.reset()
    net.eval()
    running_loss = 0.0
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False, verbose=0 if is_slurm else 1)
    skip_det = losses_params['weight'][0] == 0
    for (i, data) in enumerate(loader):
        batch_size = data[0].size(0)
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()
        seg_map_label = data[2].to('cuda').double()
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        if detection_loss != None and segmentation_loss != None:
            (classif_loss, reg_loss) = detection_loss(outputs['Detection'], label_map, losses_params)
            prediction = outputs['Segmentation'].contiguous().flatten()
            label = seg_map_label.contiguous().flatten()
            loss_seg = segmentation_loss(prediction, label)
            loss_seg *= batch_size
            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]
            loss_seg *= losses_params['weight'][2]
            loss = classif_loss + reg_loss + loss_seg
            running_loss += loss.item() * batch_size
        if check_perf:
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            labels = data[3]
            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            label_freespace = seg_map_label.detach().cpu().numpy().copy()
            for (pred_obj, pred_map, true_obj, true_map) in zip(out_obj, out_seg, labels, label_freespace):
                decoded_obj = np.asarray(encoder.decode(pred_obj, 0.05)) if not skip_det else np.array([])
                metrics.update(pred_map[0], true_map, decoded_obj, true_obj, threshold=0.2, range_min=5, range_max=100)
        kbar.update(i)
    (mAP, mAR, mIoU) = metrics.GetMetrics()
    return {'loss': running_loss / len(loader.dataset), 'mAP': mAP, 'mAR': mAR, 'mIoU': mIoU}

def run_FullEvaluation(net, loader, encoder, iou_threshold=0.5, config=None, save_path=None, epoch=None):
    net.eval()
    results = []
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)
    print('Generating Predictions...')
    predictions = {'prediction': {'objects': [], 'freespace': []}, 'label': {'objects': [], 'freespace': []}}
    for (i, data) in enumerate(loader):
        inputs = data[0].to('cuda').float()
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
        labels_object = data[3]
        label_freespace = data[2].numpy().copy()
        for (pred_obj, pred_map, true_obj, true_map) in zip(out_obj, out_seg, labels_object, label_freespace):
            predictions['prediction']['objects'].append(np.asarray(encoder.decode(pred_obj, 0.05)))
            predictions['label']['objects'].append(true_obj)
            predictions['prediction']['freespace'].append(pred_map[0])
            predictions['label']['freespace'].append(true_map)
        kbar.update(i)
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for iou_ in iou_list:
        results.append(GetFullMetrics(predictions['prediction']['objects'], predictions['label']['objects'], range_min=5, range_max=100, IOU_threshold=iou_))
    mIoU = []
    for i in range(len(predictions['prediction']['freespace'])):
        pred = predictions['prediction']['freespace'][i][:124].reshape(-1) >= 0.5
        label = predictions['label']['freespace'][i][:124].reshape(-1)
        intersection = np.abs(pred * label).sum()
        union = np.sum(label) + np.sum(pred) - intersection
        iou = intersection / union
        mIoU.append(iou)
    mIoU = np.asarray(mIoU).mean()
    print('------- Freespace Scores ------------')
    print('  mIoU', mIoU * 100, '%')
    detection_results = {}
    for (IOU_threshold, precision, recall, F1, range_error, angle_error) in results:
        detection_results[IOU_threshold] = {'Precision': precision, 'Recall': recall, 'F1': F1, 'Range Error': range_error, 'Angle Error': angle_error}
    results_dict = {'epoch': epoch, 'detection': detection_results, 'freespace': {'mIoU': mIoU}}
    save_results_json(results_dict, save_path)

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for (k, v) in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj

def save_results_json(results, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
