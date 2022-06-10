import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/LA/', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='SSNet', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=4, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path
snapshot_path = "./model/LA_{}_{}_labeled/{}".format(args.exp, args.labelnum, args.model)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    prototype_memory = feature_memory.FeatureMemory(elements_per_class=32, n_classes=num_classes)
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    dice_loss = losses.Binary_dice_loss
    adv_loss=losses.VAT3d(epi=args.magnitude)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            outputs, embedding = model(volume_batch)
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
            labeled_features = labeled_features.permute(0, 2, 3, 4, 1)
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


            unlabeled_features = unlabeled_features.permute(0, 2, 3, 4, 1).reshape(-1, labeled_features.size()[-1])
            pseudo_label = pseudo_label.reshape(-1)

            # get predicted features
            proj_feat_unlabeled = model.projection_head(unlabeled_features)
            pred_feat_unlabeled = model.prediction_head(proj_feat_unlabeled)

            # Apply contrastive learning loss
            loss_contr_unlabeled = contrastive_losses.contrastive_class_to_class_learned_memory(model, pred_feat_unlabeled, pseudo_label, num_classes, prototype_memory.memory)

            loss_seg = F.cross_entropy(outputs[:args.labeled_bs], true_labels)

            loss_seg_dice = dice_loss(y[:,1,...], (true_labels == 1))
            
            loss_lds = adv_loss(model, volume_batch)
            
            iter_num = iter_num + 1

            writer.add_scalar('1_Loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('1_Loss/loss_ce', loss_seg, iter_num)
            writer.add_scalar('1_Loss/loss_lds', loss_lds, iter_num)
            writer.add_scalar('1_Loss/loss_cl_l', loss_contr_labeled, iter_num)
            writer.add_scalar('1_Loss/loss_cl_u', loss_contr_unlabeled, iter_num)

            dice_all = metric.binary.dc((y[:,1,...] > 0.5).cpu().data.numpy(), label_batch[:args.labeled_bs,...].cpu().data.numpy())
            
            writer.add_scalar('2_Dice/Dice_all', dice_all, iter_num)
            
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            loss =  loss_seg_dice + consistency_weight * (loss_lds + 0.1 * (loss_contr_labeled + loss_contr_unlabeled))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_dice: %03f, loss_lds: %03f, loss_cl_l: %03f, loss_cl_u: %03f' % (iter_num, loss, loss_seg_dice, loss_lds, loss_contr_labeled, loss_contr_unlabeled))
            writer.add_scalar('3_consist_weight', consistency_weight, iter_num)

            if iter_num >= 800 and iter_num % 200 == 0:
                ins_width = 2
                B,C,H,W,D = outputs.size()
                snapshot_img = torch.zeros(size = (D, 3, 3*H + 3 * ins_width, W + ins_width), dtype = torch.float32)

                snapshot_img[:,:, H:H+ ins_width,:] = 1
                snapshot_img[:,:, 2*H + ins_width:2*H + 2*ins_width,:] = 1
                snapshot_img[:,:, 3*H + 2*ins_width:3*H + 3*ins_width,:] = 1
                snapshot_img[:,:, :,W:W+ins_width] = 1

                seg_out = outputs_soft[args.labeled_bs,1,...].permute(2,0,1) # y
                target =  label_batch[args.labeled_bs,...].permute(2,0,1)
                train_img = volume_batch[args.labeled_bs,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                
                writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch_num, iter_num), snapshot_img)

                seg_out = outputs_soft[0,1,...].permute(2,0,1) # y
                target =  label_batch[0,...].permute(2,0,1)
                train_img = volume_batch[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_label'% (epoch_num, iter_num), snapshot_img)
            
            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= 2000 and iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()


            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
