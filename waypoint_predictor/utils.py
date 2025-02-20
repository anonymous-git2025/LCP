
import torch
import numpy as np
import sys
import glob
import json
import copy

def neighborhoods(mu, x_range, y_range, sigma, circular_x=True, gaussian=False):
    
    x_mu = mu[:,0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:,1].unsqueeze(1).unsqueeze(1)

    
    x = torch.arange(start=0,end=x_range, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
    y = torch.arange(start=0,end=y_range, device=mu.device, dtype=mu.dtype).unsqueeze(1).unsqueeze(0)

    y_diff = y - y_mu
    x_diff = x - x_mu
    if circular_x:
        x_diff = torch.min(torch.abs(x_diff), torch.abs(x_diff + x_range))
    if gaussian:
        output = torch.exp(-0.5 * ((x_diff/sigma[0])**2 + (y_diff/sigma[1])**2 ))
    else:
        output = torch.logical_and(
            torch.abs(x_diff) <= sigma[0], torch.abs(y_diff) <= sigma[1]
        ).type(mu.dtype)

    return output


def nms(pred, max_predictions=10, sigma=(1.0,1.0), gaussian=False):
    

    shape = pred.shape

    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0],-1))  
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0],-1))  

    for i in range(max_predictions):
        
        flat_supp_pred = supp_pred.reshape((shape[0],-1))
        val, ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0,shape[0])
        flat_output[indices,ix] = flat_pred[indices,ix]

      
        y = ix / shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x,y], dim=1).float()

        g = neighborhoods(mu, shape[-1], shape[-2], sigma, gaussian=gaussian)

        supp_pred *= (1-g.unsqueeze(1))

    output[output < 0] = 0
    return output


def get_gt_nav_map(num_angles, nav_dict, scan_ids, waypoint_ids):
    
    
    bs = len(scan_ids)      
    target = torch.zeros(bs, num_angles, 12)       
    obstacle = torch.zeros(bs, num_angles, 12)
    weight = torch.zeros(bs, num_angles, 12)
    source_pos = []
    target_pos = []

    for i in range(bs):
        target[i] = torch.tensor(nav_dict[scan_ids[i]][waypoint_ids[i]]['target'])
        obstacle[i] = torch.tensor(nav_dict[scan_ids[i]][waypoint_ids[i]]['obstacle'])
        weight[i] = torch.tensor(nav_dict[scan_ids[i]][waypoint_ids[i]]['weight'])
        source_pos.append(nav_dict[scan_ids[i]][waypoint_ids[i]]['source_pos'])
        target_pos.append(nav_dict[scan_ids[i]][waypoint_ids[i]]['target_pos'])

    return target, obstacle, weight, source_pos, target_pos


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=10):
   
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def save_checkpoint(epoch, net, net_optimizer, path, best_val_1, best_val_2):
    ''' Snapshot models '''
    states = {}
    def create_state(name, model, optimizer):
        states[name] = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_1': copy.deepcopy(best_val_1),
            'best_val_2': copy.deepcopy(best_val_2)
        }
    all_tuple = [("predictor", net, net_optimizer)]
    for param in all_tuple:
        create_state(*param)
    torch.save(states, path)


def load_checkpoint(net, net_optimizer, path):
    ''' Loads parameters (but not training state) '''
    states = torch.load(path)
    def recover_state(name, model, optimizer):
        state = model.state_dict()
        model_keys = set(state.keys())
        load_keys = set(states[name]['state_dict'].keys())
        if model_keys != load_keys:
            print("NOTICE: DIFFERENT KEYS FOUND")
        state.update(states[name]['state_dict'])
        model.load_state_dict(state)
        optimizer.load_state_dict(states[name]['optimizer'])
    all_tuple = [("predictor", net, net_optimizer)]
    for param in all_tuple:
        recover_state(*param)

    if 'best_val_1' in states['predictor'].keys():
        best_val_1 = states['predictor']['best_val_1']
    else:
        best_val_1 = None
    if 'best_val_2' in states['predictor'].keys():
        best_val_2 = states['predictor']['best_val_2']
    else:
        best_val_2 = None

    return states['predictor']['epoch'], all_tuple[0][1], all_tuple[0][2], copy.deepcopy(best_val_1), copy.deepcopy(best_val_2)


def get_attention_mask(num_imgs=24, neighbor=2):
    assert neighbor <= 5

    mask = np.zeros((num_imgs,num_imgs))
    t = np.zeros(num_imgs)
    t[:neighbor+1] = np.ones(neighbor+1)       
    if neighbor != 0:
        t[-neighbor:] = np.ones(neighbor)       
    for ri in range(num_imgs):
        mask[ri] = t           
        t = np.roll(t, 1)       

    return torch.from_numpy(mask).reshape(1,1,num_imgs,num_imgs).long()


def load_gt_navigability(path):
    ''' waypoint ground-truths '''
    all_scans_nav_map = {}         
    gt_dir = glob.glob('%s*'%(path))       
    for gt_dir_i in gt_dir:    
        with open(gt_dir_i, 'r') as f:
            nav_map = json.load(f)
        for scan_id, values in nav_map.items():
            all_scans_nav_map[scan_id] = values    
    return all_scans_nav_map

def get_gt_nav_map_my_data(num_angles, nav_dict, scan_ids, episode_ids, step_ids, waypoint_ids):
    
  
    bs = len(scan_ids)     
    target = torch.zeros(bs, num_angles, 12)       
    obstacle = torch.zeros(bs, num_angles, 12)
    weight = torch.zeros(bs, num_angles, 12)
    source_pos = []
    target_pos = []

    for i in range(bs):
        target[i] = torch.tensor(nav_dict[scan_ids[i]][episode_ids[i]][int(step_ids[i])][waypoint_ids[i]]['target'])
        obstacle[i] = torch.tensor(nav_dict[scan_ids[i]][episode_ids[i]][int(step_ids[i])][waypoint_ids[i]]['obstacle'])
        weight[i] = torch.tensor(nav_dict[scan_ids[i]][episode_ids[i]][int(step_ids[i])][waypoint_ids[i]]['weight'])
        source_pos.append(nav_dict[scan_ids[i]][episode_ids[i]][int(step_ids[i])][waypoint_ids[i]]['source_pos'])
        target_pos.append(nav_dict[scan_ids[i]][episode_ids[i]][int(step_ids[i])][waypoint_ids[i]]['target_pos'])

    return target, obstacle, weight, source_pos, target_pos

def load_from_pretrained(model, path):
    save_model = torch.load(path)['predictor']['state_dict']
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)  
    model.load_state_dict(model_dict)

    return model
