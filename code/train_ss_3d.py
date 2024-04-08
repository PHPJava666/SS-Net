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

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")#model是VNet
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
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)#两路采样器，一路是有标签的样本，一路是无标签的样本，分的很清楚
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    dice_loss = losses.Binary_dice_loss
    adv_loss=losses.VAT3d(epi=args.magnitude)#ssnet特有
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
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']#volume_batch和label_batch数量一致
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()#使用VNet进行分割
            outputs, embedding = model(volume_batch)#outputs是分割结果，即模型对每个输入图像的预测结果，embedding是特征，即模型的内部表示，通常用于进一步的处理或分析。这里是把一个batch_size里面的都分割了
            outputs_soft = F.softmax(outputs, dim=1)#对输出的output进行softmax处理，dim=1表示对每一行进行softmax处理
            labeled_features = embedding[:args.labeled_bs,...]#这两行代码将嵌入embedding分成两部分：label_features是有标签的样本的特征
            unlabeled_features = embedding[args.labeled_bs:,...]#unlabeled_features是无标签的样本的特征
            y = outputs_soft[:args.labeled_bs]#y是有标签的样本的预测结果
            true_labels = label_batch[:args.labeled_bs]#true_labels是有标签的样本的真实标签

            #labeled_bs， default=2, help='batch_size of labeled data per gpu')
            #batch_size， default=4, help='batch_size per gpu')每个batch有4个样本，其中有2个有标签，2个无标签
            
            _, prediction_label = torch.max(y, dim=1)#获取模型对有标签样本的预测结果，y是模型输出，是一个概率分布。torch.max(y, dim=1)返回y中每一行的最大值和最大值的索引，即预测的标签,这里只取了索引，即预测的标签
            _, pseudo_label = torch.max(outputs_soft[args.labeled_bs:], dim=1)  # Get pseudolabels
            #这行代码是获取模型对无标签样本的预测结果，即伪标签。
            
            mask_prediction_correctly = ((prediction_label == true_labels).float() * (prediction_label > 0).float()).bool()
	        ### select the correct predictions and ignore the background class
            #这行代码是在创建一个掩码，用于标记预测正确的样本。这里首先比较预测的标签prediction_label和真实标签true_labels，相等的地方会得到True，不等的地方会得到False。然后，检查预测的标签是否大于0，因为在这个上下文中，0可能被用作背景类或无效类。最后，将结果转换为布尔类型，得到的mask_prediction_correctly可以用于后续的计算，例如计算预测的准确率。

            # Apply the filter mask to the features and its labels这段代码用于处理模型的特征输出，并从中选择预测正确的样本。
            labeled_features = labeled_features.permute(0, 2, 3, 4, 1)#这里首先将特征张量labeled_features和标签张量true_labels的维度进行转换，以便与掩码mask_prediction_correctly进行匹配。
            labels_correct = true_labels[mask_prediction_correctly]#然后，使用掩码mask_prediction_correctly选择预测正确的样本。
            labeled_features_correct = labeled_features[mask_prediction_correctly, ...]#最后，将选择的特征和标签存储在labeled_features_correct和labels_correct中。

            # get projected features这段代码主要用于在模型的评估模式下，对预测正确的样本的特征进行投影，并在完成后将模型切换回训练模式。
            with torch.no_grad():#这行代码是是在创建一个不需要计算梯度的上下文。在这个上下文中，多以后的计算都不会跟踪梯度，这可以减少内存的使用，并加速计算。这在评估模型或者进行推力时是非常有用的，因为在这些情况下通常是不需要计算梯度的。
                model.eval()#这里是为了防止在更新内存时，使用的特征是更新后的特征，而不是更新前的特征。因此，这里将模型设置为评估模式，以便在更新内存时，使用的特征是更新前的特征。
                #这行代码是将模型设置为评估模式。在pytorch中，模型有两种模式：训练模式和评估模式。在评估模式下，模型的某些层（如Dropout或BatchNorm）会以不同的方式工作，以适应模型的评估。
                proj_labeled_features_correct = model.projection_head(labeled_features_correct)#这里是将选择的特征labeled_features_correct通过模型的投影头projection_head，得到投影后的特征proj_labeled_features_correct。
                #这行代码是对预测正确的样本的特征label_features_correct进行投影.这里的model.projection_head是模型的投影头，用于将特征映射到一个更高维度的空间。这里的proj_labeled_features_correct是投影后的特征。
                model.train()#这里将模型设置为训练模式，以便在训练时，使用的特征是更新后的特征。

            # updated memory bank 这段代码主要用于更新原型记忆库，并对特征和标签进行重塑
            prototype_memory.add_features_from_sample_learned(model, proj_labeled_features_correct, labels_correct)#这段代码时调用prototype_memory的add_features_from_sample_learned方法，将预测正确的样本的投影特征proj_labeled_features_correct和对应的标签labels_correct添加到原型记忆库中。在这个上下文中，pro_memory可能是一个用于存储每个类别的原型特征的数据结构，add_features_from_sample_learned方法用于更新这些原型特征。
            labeled_features_all = labeled_features.reshape(-1, labeled_features.size()[-1])#这行代码是将特征张量labeled_features的维度进行重塑，以便与标签张量true_labels进行匹配。
            #这行代码是在将label_features重塑为二维张量。
            labeled_labels = true_labels.reshape(-1)#这行代码是在将true_labels重塑为一维张量。reshape(-1)会将true_labels的形状改变为（-1，）,也就是一个一维张量。这样，label_labels就会是一个长度为num_sample的一维张量，其中每个元素对应一个样本的标签。

            # get predicted features这段代码主要用于对所有标签样本的特征进行投影和预测。
            proj_labeled_features_all = model.projection_head(labeled_features_all)#这行代码是将所有标签样本的特征labeled_features_all通过模型的投影头projection_head，得到投影后的特征proj_labeled_features_all。
            #这行代码是在对所有标签样本的特征label_features_all进行投影。这里的model.projection_head可能是一个神经网络，用于将特征投射到一个新的特征空间。这在一些算法中是常见的操作，例如在对比学习中，通常会有一个投影头用于将特征投影到一个新的特征空间，以便进行对比。
            pred_labeled_features_all = model.prediction_head(proj_labeled_features_all)#这行代码是将投影后的特征proj_labeled_features_all通过模型的预测头prediction_head，得到预测后的特征pred_labeled_features_all。
            #这行代码是对投影后的特征proj_labeled_features_all进行预测。这里的model.prediction_head可能是另一个神经网络，用于对投影后的特征进行预测，得到预测结果pred_labeled_features_all。这个预测结果可能会用于计算损失，以便在反向传播过程中更新模型的参数。

            # Apply contrastive learning loss
            loss_contr_labeled = contrastive_losses.contrastive_class_to_class_learned_memory(model, pred_labeled_features_all, labeled_labels, num_classes, prototype_memory.memory)#这行代码是在计算对比损失。对比损失是一种常用于自监督学习和半监督学习的损失函数，它鼓励模型将相同类别的样本映射到接近的特征向量，将不同类别的样本映射到远离的特征向量。
            #这里的consistency_losses.contrastive_class_to_class_learned_memory函数接收五个参数：模型model,预测的特征pred_labeled_features_all, 对应的标签label_labels,类别数量num_classes和原型记忆库prototype_memory.memory。            
            """
            来自SemiSeg-Contrastive的对比损失函数
            More details can be checked at https://github.com/Shathe/SemiSeg-Contrastive
            """


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
