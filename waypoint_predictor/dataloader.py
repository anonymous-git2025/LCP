
import glob
import numpy as np
from PIL import Image
import pickle as pkl

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class RGBDepthPano(Dataset):
    def __init__(self, args, img_dir, navigability_dict):
        self.RGB_INPUT_DIM = 224
        self.DEPTH_INPUT_DIM = 256
        self.NUM_IMGS = args.NUM_IMGS
        self.navigability_dict = navigability_dict

        self.rgb_transform = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            )

        self.img_dirs = glob.glob(img_dir)      

        for img_dir in glob.glob(img_dir):
            scan_id = img_dir.split('/')[-1][:11]       
            waypoint_id = img_dir.split('/')[-1][12:-14]       
            if waypoint_id not in self.navigability_dict[scan_id]:     
                self.img_dirs.remove(img_dir)      

    def __len__(self): 
        return len(self.img_dirs)   

    def __getitem__(self, idx):

        img_dir = self.img_dirs[idx]
        sample_id = str(idx)
        scan_id = img_dir.split('/')[-1][:11]
        waypoint_id = img_dir.split('/')[-1][12:-14]

        ''' rgb and depth images '''
        rgb_depth_img = pkl.load(open(img_dir, "rb"))
        rgb_img = torch.from_numpy(rgb_depth_img['rgb']).permute(0, 3, 1, 2)
        depth_img = torch.from_numpy(rgb_depth_img['depth']).permute(0, 3, 1, 2)

       
        trans_rgb_imgs = torch.zeros(self.NUM_IMGS, 3, self.RGB_INPUT_DIM, self.RGB_INPUT_DIM)
        trans_depth_imgs = torch.zeros(self.NUM_IMGS, self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)

        no_trans_rgb = torch.zeros(self.NUM_IMGS, 3, self.RGB_INPUT_DIM, self.RGB_INPUT_DIM, dtype=torch.uint8)
        no_trans_depth = torch.zeros(self.NUM_IMGS, self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)

        for ix in range(self.NUM_IMGS):
            trans_rgb_imgs[ix] = self.rgb_transform(rgb_img[ix])        
            trans_depth_imgs[ix] = depth_img[ix][0]     


        sample = {'sample_id': sample_id,
                  'scan_id': scan_id,
                  'waypoint_id': waypoint_id,
                  'rgb': trans_rgb_imgs,
                  'depth': trans_depth_imgs.unsqueeze(-1),  
                  }

        return sample
    
class MyDataLoader(Dataset):
    def __init__(self, args, img_dir, navigability_dict):
        self.RGB_INPUT_DIM = 224
        self.DEPTH_INPUT_DIM = 256
        self.NUM_IMGS = args.NUM_IMGS
        self.navigability_dict = navigability_dict

        self.rgb_transform = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            )
        

        self.img_dirs = glob.glob(img_dir)      

        for img_dir in glob.glob(img_dir):
            file_name = img_dir.split('/')[-1]
            scan_id = file_name.split('_')[0]       
            episode_id = file_name.split('_')[1]
            step_id = file_name.split('_')[2]
            
            if self.navigability_dict[scan_id][episode_id][int(step_id)] == None:     
                self.img_dirs.remove(img_dir)     

    def __len__(self): 
        return len(self.img_dirs)      
   
    def __getitem__(self, idx): 

        img_dir = self.img_dirs[idx]
        sample_id = str(idx)
        file_name = img_dir.split('/')[-1]
        scan_id = file_name.split('_')[0]
        episode_id = file_name.split('_')[1]
        step_id = file_name.split('_')[2]
        if len(file_name.split('_')) > 4:
            node_id = '_'.join(file_name.split('_')[3:])[:-4]
        else:
            node_id = file_name.split('_')[3][:-4]

        ''' rgb and depth images '''
        rgb_depth_img = pkl.load(open(img_dir, "rb"))
        rgb_img = torch.from_numpy(rgb_depth_img['rgb']).permute(0, 3, 1, 2)
        depth_img = torch.from_numpy(rgb_depth_img['depth']).permute(0, 3, 1, 2)
        
        trans_rgb_imgs = torch.zeros(self.NUM_IMGS, 3, self.RGB_INPUT_DIM, self.RGB_INPUT_DIM)
        trans_depth_imgs = torch.zeros(self.NUM_IMGS, self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)

       
        for ix in range(self.NUM_IMGS):
            trans_rgb_imgs[ix] = self.rgb_transform(rgb_img[ix])        
            trans_depth_imgs[ix] = depth_img[ix][0]    
            

        sample = {'sample_id': sample_id,
                  'scan_id': scan_id,
                  'episode_id': episode_id,
                  'step_id': step_id,
                  'waypoint_id': node_id,
                  'rgb': trans_rgb_imgs,
                  'depth': trans_depth_imgs.unsqueeze(-1),  
                  }


        return sample