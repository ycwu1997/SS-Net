import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SSNet', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')

args = parser.parse_args()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, 
                            split="train", 
                            num=None, 
                            transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    prototype_memory = feature_memory.FeatureMemory(elements_per_class=32, n_classes=num_classes)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(n_classes=num_classes)
    adv_loss=losses.VAT2d(epi=args.magnitude)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, embedding= model(volume_batch)
            outputs_soft = F.softmax(outputs, dim=1)

            labeled_features = embedding[:args.labeled_bs,...]
            unlabeled_features = embedding[args.labeled_bs:,...]
            y = outputs_soft[:args.labeled_bs]
            true_labels = label_batch[:args.labeled_bs]
            
            _, prediction_label = torch.max(y, dim=1)
            _, pseudo_label = torch.max(outputs_soft[args.labeled_bs:], dim=1)  # Get pseudolabels
            
            mask_prediction_correctly = ((prediction_label == true_labels).float() * (prediction_label > 0).float()).bool()
            ### select the correct predictions and ignore the background class
                        
            # Apply the filter mask to the features and its labels
            labeled_features = labeled_features.permute(0, 2, 3, 1)
            labels_correct = true_labels[mask_prediction_correctly]
            labeled_features_correct = labeled_features[mask_prediction_correctly, ...]

            # get projected features
            with torch.no_grad():
                model.eval()
                proj_labeled_features_correct = model.projection_head(labeled_features_correct)
                model.train()
  
            # updated memory bank
            prototype_memory.add_features_from_sample_learned(model, proj_labeled_features_correct, labels_correct)

            labeled_features_all = labeled_features.reshape(-1, labeled_features.size()[-1])
            labeled_labels = true_labels.reshape(-1)

            # get predicted features
            proj_labeled_features_all = model.projection_head(labeled_features_all)
            pred_labeled_features_all = model.prediction_head(proj_labeled_features_all)

            # Apply contrastive learning loss
            loss_contr_labeled = contrastive_losses.contrastive_class_to_class_learned_memory(model, pred_labeled_features_all, labeled_labels, num_classes, prototype_memory.memory)

            unlabeled_features = unlabeled_features.permute(0, 2, 3, 1).reshape(-1, labeled_features.size()[-1])
            pseudo_label = pseudo_label.reshape(-1)

            # get predicted features
            proj_feat_unlabeled = model.projection_head(unlabeled_features)
            pred_feat_unlabeled = model.prediction_head(proj_feat_unlabeled)

            # Apply contrastive learning loss
            loss_contr_unlabeled = contrastive_losses.contrastive_class_to_class_learned_memory(model, pred_feat_unlabeled, pseudo_label, num_classes, prototype_memory.memory)
            
            loss_seg_ce = ce_loss(outputs[:args.labeled_bs], true_labels[:].long())

            loss_seg_dice = dice_loss(y, true_labels.unsqueeze(1))

            loss_lds =adv_loss(model, volume_batch)
            
            consistency_weight = get_current_consistency_weight(iter_num//150)
            loss = loss_seg_dice + consistency_weight * (loss_lds + 0.1 * (loss_contr_labeled + loss_contr_unlabeled))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_seg_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('info/loss_vat', loss_lds, iter_num)
            writer.add_scalar('info/loss_cl_l', loss_contr_labeled, iter_num)
            writer.add_scalar('info/loss_cl_u', loss_contr_unlabeled, iter_num)
            writer.add_scalar('info/consistency_weight',    consistency_weight, iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f,loss_vat: %f, loss_cl_l: %f,loss_cl_u: %f' %(iter_num, loss, loss_seg_ce, loss_seg_dice, loss_lds, loss_contr_labeled, loss_contr_unlabeled))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/ACDC_{}_{}_labeled/{}".format(args.exp, args.labelnum, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code',shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
