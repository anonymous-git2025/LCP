
import torch
import argparse
from dataloader_lang import RGBDepthPano, MyDataLoader

from image_encoders import RGBEncoder, DepthEncoder
from TRM_net_lang import BinaryDistPredictor_TRM, TRM_predict

from eval import waypoint_eval

import os
import glob
import utils
import random
from utils import nms
from utils import print_progress
from tensorboardX import SummaryWriter

import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup(args):
    torch.manual_seed(0)
    random.seed(0)
    exp_log_path = './checkpoints/%s/'%(args.EXP_ID)
    os.makedirs(exp_log_path, exist_ok=True)
    exp_log_path = './checkpoints/%s/snap/'%(args.EXP_ID)
    os.makedirs(exp_log_path, exist_ok=True)

class Param():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train waypoint predictor')

        self.parser.add_argument('--EXP_ID', type=str, default='test_0')
        self.parser.add_argument('--TRAINEVAL', type=str, default='train', help='trian or eval mode')
        self.parser.add_argument('--VIS', type=int, default=0, help='visualize predicted hearmaps')
     

        self.parser.add_argument('--ANGLES', type=int, default=24)
        self.parser.add_argument('--NUM_IMGS', type=int, default=24)
        self.parser.add_argument('--NUM_CLASSES', type=int, default=12)
        self.parser.add_argument('--MAX_NUM_CANDIDATES', type=int, default=5)

        self.parser.add_argument('--PREDICTOR_NET', type=str, default='TRM', help='TRM only')

        self.parser.add_argument('--EPOCH', type=int, default=10)
        self.parser.add_argument('--BATCH_SIZE', type=int, default=2)
        self.parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
        self.parser.add_argument('--WEIGHT', type=int, default=0, help='weight the target map')

        self.parser.add_argument('--TRM_LAYER', default=2, type=int, help='number of TRM hidden layers')
        self.parser.add_argument('--TRM_NEIGHBOR', default=2, type=int, help='number of attention mask neighbor')
        self.parser.add_argument('--HEATMAP_OFFSET', default=2, type=int, help='an offset determined by image FoV and number of images')
        self.parser.add_argument('--HIDDEN_DIM', default=768, type=int)

        self.parser.add_argument('--load_from_ckpt', default=None, type=str)

        self.parser.add_argument('--lang', default=None, type=str, help='h_t or lang_state_scores or atteneded lang')
        self.parser.add_argument('--from_pretrained', default=None, type=str, help='h_t or lang_state_scores or atteneded lang')

        self.args = self.parser.parse_args()

def predict_waypoints(args):

    print('\nArguments', args)
    log_dir = './checkpoints/%s/tensorboard/'%(args.EXP_ID)
    writer = SummaryWriter(log_dir=log_dir)

    ''' networks '''
    rgb_encoder = RGBEncoder(resnet_pretrain=True, trainable=False).to(device)     
    depth_encoder = DepthEncoder(resnet_pretrain=True, trainable=False).to(device)
    if args.PREDICTOR_NET == 'TRM':    
        print('\nUsing TRM predictor')
        print('HIDDEN_DIM default to 768')
        args.HIDDEN_DIM = 768
        predictor = BinaryDistPredictor_TRM(args=args,
            hidden_dim=args.HIDDEN_DIM, n_classes=args.NUM_CLASSES).to(device)

    
    
    ''' load navigability (gt waypoints, obstacles and weights) '''     
    navigability_dict_my_data = utils.load_gt_navigability(     
    './adapt_collected_data/training_data/%s_*_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'%(args.ANGLES))
 
    ''' dataloader for rgb and depth images '''
    train_img_dir = './data_collect/train/*/*.pkl'
    
    traindataloader = MyDataLoader(args, train_img_dir, navigability_dict_my_data)     
    
    eval_img_dir = './data_collect/val_unseen/*/*.pkl'
    evaldataloader = MyDataLoader(args, eval_img_dir, navigability_dict_my_data)


    if args.TRAINEVAL == 'train':      
        trainloader = torch.utils.data.DataLoader(traindataloader, 
        batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4)
    evalloader = torch.utils.data.DataLoader(evaldataloader, 
        batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4)

    ''' optimization '''       
    criterion_bcel = torch.nn.BCEWithLogitsLoss(reduction='none')      
    criterion_mse = torch.nn.MSELoss(reduction='none')      

    params = list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.LEARNING_RATE)

    ''' training loop '''
    if args.TRAINEVAL == 'train':
        print('\nTraining starts')
        best_val_1 = {"avg_wayscore": 0.0, "log_string": '', "update":False}
        best_val_2 = {"avg_pred_distance": 10.0, "log_string": '', "update":False}

        start_epoch = -1
        if args.load_from_ckpt:
            print('\nLoading from:', args.load_from_ckpt)
            start_epoch, predictor, optimizer, temp_val_1, temp_val_2 = utils.load_checkpoint(
                            predictor, optimizer, args.load_from_ckpt)
            if temp_val_1:
                best_val_1 = copy.deepcopy(temp_val_1)
            if temp_val_2:
                best_val_2 = copy.deepcopy(temp_val_2)

        if args.from_pretrained:
            print('\nLoading from:', args.from_pretrained)
            predictor = utils.load_from_pretrained(predictor, args.from_pretrained)

        for epoch in range(start_epoch + 1, args.EPOCH): 
            sum_loss = 0.0

            rgb_encoder.eval()
            depth_encoder.eval()
            predictor.train()

            for i, data in enumerate(trainloader):     
                scan_ids = data['scan_id']      
                waypoint_ids = data['waypoint_id']
                episode_ids = data['episode_id']
                step_ids = data['step_id']
                rgb_imgs = data['rgb'].to(device)
                depth_imgs = data['depth'].to(device)
                lang = data['lang'].to(device)


                ''' processing observations '''
                rgb_feats = rgb_encoder(rgb_imgs)        
                depth_feats = depth_encoder(depth_imgs)  

                ''' learning objectives '''
                
                target, obstacle, weight, _, _ = utils.get_gt_nav_map_my_data(      
                    args.ANGLES, navigability_dict_my_data, scan_ids, episode_ids, step_ids, waypoint_ids)
            
                target = target.to(device)
                obstacle = obstacle.to(device)
                weight = weight.to(device)

                if args.PREDICTOR_NET == 'TRM':
                    vis_logits = TRM_predict('train', args,
                        predictor, rgb_feats, depth_feats, lang)

                    loss_vis = criterion_mse(vis_logits, target)        
                    if args.WEIGHT:     
                        loss_vis = loss_vis * weight
                    total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES      
                total_loss.backward()
                optimizer.step()
                sum_loss += total_loss.item()

                print_progress(i+1, len(trainloader), prefix='Epoch: %d/%d'%((epoch+1),args.EPOCH))
            writer.add_scalar("Train/Loss", sum_loss/(i+1), epoch)     
            print('Train Loss: %.5f' % (sum_loss/(i+1)))  

            ''' evaluation - inference '''
            
            sum_loss = 0.0
            predictions = {'sample_id': [], 
                'source_pos': [], 'target_pos': [],
                'probs': [], 'logits': [],
                'target': [], 'obstacle': [], 'sample_loss': []}

            rgb_encoder.eval()
            depth_encoder.eval()
            predictor.eval()

            for i, data in enumerate(evalloader):      
                scan_ids = data['scan_id']
                waypoint_ids = data['waypoint_id']
                
                episode_ids = data['episode_id']
                step_ids = data['step_id']
                sample_id = data['sample_id']
                rgb_imgs = data['rgb'].to(device)
                depth_imgs = data['depth'].to(device)

                lang = data['lang'].to(device)

                
                target, obstacle, weight, \
                source_pos, target_pos = utils.get_gt_nav_map_my_data(      
                    args.ANGLES, navigability_dict_my_data, scan_ids, episode_ids, step_ids, waypoint_ids)
            
                    
                target = target.to(device)
                obstacle = obstacle.to(device)
                weight = weight.to(device)

                ''' processing observations '''
                rgb_feats = rgb_encoder(rgb_imgs)        
                depth_feats = depth_encoder(depth_imgs)  

                if args.PREDICTOR_NET == 'TRM':
                    vis_probs, vis_logits = TRM_predict('eval', args,
                        predictor, rgb_feats, depth_feats, lang)
                    overall_probs = vis_probs       
                    overall_logits = vis_logits
                    loss_vis = criterion_mse(vis_logits, target)
                    if args.WEIGHT:
                        loss_vis = loss_vis * weight
                    sample_loss = loss_vis.sum(-1).sum(-1) / args.ANGLES
                    total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES

                sum_loss += total_loss.item()
                predictions['sample_id'].append(sample_id)
                predictions['source_pos'].append(source_pos)
                predictions['target_pos'].append(target_pos)
                predictions['probs'].append(overall_probs.tolist())
                predictions['logits'].append((overall_logits.tolist()))
                predictions['target'].append(target.tolist())
                predictions['obstacle'].append(obstacle.tolist())      
                predictions['sample_loss'].append(target.tolist())     

            print('Eval Loss: %.5f' % (sum_loss/(i+1)))
            results = waypoint_eval(args, predictions)
            writer.add_scalar("Evaluation/Loss", sum_loss/(i+1), epoch)
            writer.add_scalar("Evaluation/p_waypoint_openspace", results['p_waypoint_openspace'], epoch)
            writer.add_scalar("Evaluation/p_waypoint_obstacle", results['p_waypoint_obstacle'], epoch)
            writer.add_scalar("Evaluation/avg_wayscore", results['avg_wayscore'], epoch)       
            writer.add_scalar("Evaluation/avg_pred_distance", results['avg_pred_distance'], epoch)     
            log_string = 'Epoch %s '%(epoch)
            for key, value in results.items():
                if key != 'candidates':
                    log_string += '{} {:.5f} | '.format(str(key), value)
            print(log_string)

            
            if results['avg_wayscore'] > best_val_1['avg_wayscore']:       
                checkpoint_save_path = './checkpoints/%s/snap/check_val_best_avg_wayscore'%(args.EXP_ID) 
                best_val_1['avg_wayscore'] = results['avg_wayscore']
                best_val_1['log_string'] = log_string
                utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_save_path, best_val_1, best_val_2)
                print('New best avg_wayscore result found, checkpoint saved to %s'%(checkpoint_save_path))
           
            print('Best avg_wayscore result til now: ', best_val_1['log_string'])

            if results['avg_pred_distance'] < best_val_2['avg_pred_distance']:     
                checkpoint_save_path = './checkpoints/%s/snap/check_val_best_avg_pred_distance'%(args.EXP_ID) 
                best_val_2['avg_pred_distance'] = results['avg_pred_distance']
                best_val_2['log_string'] = log_string
                utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_save_path, best_val_1, best_val_2)
                print('New best avg_pred_distance result found, checkpoint saved to %s'%(checkpoint_save_path))
            checkpoint_reg_save_path = './checkpoints/%s/snap/check_latest'%(args.EXP_ID) 
            utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_reg_save_path, best_val_1, best_val_2)
            print('Best avg_pred_distance result til now: ', best_val_2['log_string'])

            if args.lang == 'attened_lang' and results['p_waypoint_openspace'] > 0.86:
                ckpt_save_path = './checkpoints/%s/snap/epoch'%(args.EXP_ID) + str(epoch)
                utils.save_checkpoint(epoch+1, predictor, optimizer, ckpt_save_path, best_val_1, best_val_2)

    elif args.TRAINEVAL == 'eval':
        ''' evaluation - inference (with a bit mixture-of-experts) '''
        print('\nEvaluation mode, please doublecheck EXP_ID and LOAD_EPOCH')
        
        checkpoint_load_path = 'waypoint_predictor/waypoint_predict/checkpoints/att_lang_no_pretrained_third_round_2/snap/0'
        epoch, predictor, optimizer, _, _ = utils.load_checkpoint(
                        predictor, optimizer, checkpoint_load_path)

        sum_loss = 0.0
        predictions = {'sample_id': [], 
            'source_pos': [], 'target_pos': [],
            'probs': [], 'logits': [],
            'target': [], 'obstacle': [], 'sample_loss': []}

        rgb_encoder.eval()
        depth_encoder.eval()
        predictor.eval()

        for i, data in enumerate(evalloader):
            if args.VIS and i == 5:
                break

            scan_ids = data['scan_id']
            waypoint_ids = data['waypoint_id']
            
            episode_ids = data['episode_id']
            step_ids = data['step_id']
            sample_id = data['sample_id']
            rgb_imgs = data['rgb'].to(device)
            depth_imgs = data['depth'].to(device)

            lang = data['lang'].to(device)

            
            target, obstacle, weight, \
            source_pos, target_pos = utils.get_gt_nav_map_my_data(   
                args.ANGLES, navigability_dict_my_data, scan_ids, episode_ids, step_ids, waypoint_ids)
            
            target = target.to(device)
            obstacle = obstacle.to(device)
            weight = weight.to(device)

            ''' processing observations '''
            rgb_feats = rgb_encoder(rgb_imgs)       
            depth_feats = depth_encoder(depth_imgs)  

            ''' predicting the waypoint probabilities '''
            if args.PREDICTOR_NET == 'TRM':
                vis_probs, vis_logits = TRM_predict('eval', args,
                    predictor, rgb_feats, depth_feats, lang)
                overall_probs = vis_probs
                overall_logits = vis_logits
                loss_vis = criterion_mse(vis_logits, target)

                if args.WEIGHT:
                    loss_vis = loss_vis * weight
                sample_loss = loss_vis.sum(-1).sum(-1) / args.ANGLES
                total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES

            sum_loss += total_loss.item()
            predictions['sample_id'].append(sample_id)
            predictions['source_pos'].append(source_pos)
            predictions['target_pos'].append(target_pos)
            predictions['probs'].append(overall_probs.tolist())
            predictions['logits'].append(overall_logits.tolist())
            predictions['target'].append(target.tolist())
            predictions['obstacle'].append(obstacle.tolist())
            predictions['sample_loss'].append(target.tolist())

        print('Eval Loss: %.5f' % (sum_loss/(i+1)))
        results = waypoint_eval(args, predictions)
        log_string = 'Epoch %s '%(epoch)
        for key, value in results.items():
            if key != 'candidates':
                log_string += '{} {:.5f} | '.format(str(key), value)
        print(log_string)
        print('Evaluation Done')

    else:
        RunningModeError

if __name__ == "__main__":
    param = Param()
    args = param.args
    setup(args)

    if args.VIS:        
        assert args.TRAINEVAL == 'eval'

    predict_waypoints(args)
