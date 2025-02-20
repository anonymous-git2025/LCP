import gc
import os
import random
import warnings
from collections import defaultdict

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations
from .models.utils import (
    length2mask, dir_angle_feature_with_ele,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  

import torch.distributed as distr
import gzip
import json
from copy import deepcopy

@baseline_registry.register_trainer(name="schedulesampler-VLNBERT")
class SSTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) 

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def save_checkpoint(self, epoch: int, step_id: int) -> None:
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "epoch": epoch,
                "step_id": step_id,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.{epoch}.pth"),
        )

    def allocate_allowed_episode_by_scene(self):
        
        with gzip.open(
            self.config.TASK_CONFIG.DATASET.DATA_PATH.format(
                split=self.split)
        ) as f:
            data = json.load(f) 

        
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=self.split)
        ) as f:
            gt_data = json.load(f)

        data = data['episodes']
        
        long_episode_ids = []
        average_length = (len(data) - len(long_episode_ids))//self.world_size

        episodes_by_scene = {}
        for ep in data:
            scan = ep['scene_id'].split('/')[1]
            if scan not in episodes_by_scene.keys():
                episodes_by_scene[scan] = []
            if ep['episode_id'] not in long_episode_ids:
                episodes_by_scene[scan].append(ep['episode_id'])
            else:
                continue

        
        values_to_scenes = {}
        values = []
        for k,v in episodes_by_scene.items():
            values.append(len(v))
            if len(v) not in values_to_scenes.keys():
                values_to_scenes[len(v)] = []
            values_to_scenes[len(v)].append(k)

        groups = self.world_size
        values.sort(reverse=True)
        last_scene_episodes = episodes_by_scene[values_to_scenes[values[0]].pop()]
        values = values[1:]

        load_balance_groups = [[] for grp in range(groups)]
        scenes_groups = [[] for grp in range(groups)]

        for v in values:
            current_total = [sum(grp) for grp in load_balance_groups]
            min_index = np.argmin(current_total)
            load_balance_groups[min_index].append(v)
            scenes_groups[min_index] += episodes_by_scene[values_to_scenes[v].pop()]

        for grp in scenes_groups:
            add_number = average_length - len(grp)
            grp += last_scene_episodes[:add_number]
            last_scene_episodes = last_scene_episodes[add_number:]

        return scenes_groups[self.local_rank]

    def allocate_allowed_episode_by_scene_random(self):
        
        with gzip.open(
            self.config.TASK_CONFIG.DATASET.DATA_PATH.format(
                split=self.split)
        ) as f:     
            data = json.load(f) 
        
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=self.split)
        ) as f:
            gt_data = json.load(f)      

        data = data['episodes']
        
        long_episode_ids = []
        average_length = (len(data) - len(long_episode_ids))//self.world_size       

        episodes_by_scene = {}      
        for ep in data:
            scan = ep['scene_id'].split('/')[1]     
            if scan not in episodes_by_scene.keys():
                episodes_by_scene[scan] = []
            if ep['episode_id'] not in long_episode_ids:
                episodes_by_scene[scan].append(ep['episode_id'])
            else:
                continue
        
        id_list = []
        for ep in data:
            id_list.append(ep['episode_id'])

        groups = self.world_size
        
        scenes_groups = [[] for grp in range(groups)]       

        for index in range(groups - 1):
            scenes_groups[index] = id_list[index * average_length : (index + 1) * average_length]
        scenes_groups[groups - 1] = id_list[(groups - 1) * average_length :]


        return scenes_groups[self.local_rank]   
    
    def train_ml(self, in_train=True, train_tf=False, train_rl=False):
        self.envs.resume_all()
        observations = self.envs.reset()

        shift_index = 0
        for i, ep in enumerate(self.envs.current_episodes()):
            if ep.episode_id in self.trained_episodes:
                i = i - shift_index
                observations.pop(i)
                self.envs.pause_at(i)
                shift_index += 1
                if self.envs.num_envs == 0:
                    break
            else:
                self.trained_episodes.append(ep.episode_id)

        if self.envs.num_envs == 0:
            return -1

        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        

        not_done_masks = torch.zeros(
            self.envs.num_envs, 1, dtype=torch.bool, device=self.device)
        ml_loss = 0.
        total_weight = 0.
        losses = []
        not_done_index = list(range(self.envs.num_envs))

        
        if 'VLNBERT' in self.config.MODEL.policy_name:
            lang_idx_tokens = batch['instruction']
            padding_idx = 0
            all_lang_masks = (lang_idx_tokens != padding_idx)
            lang_lengths = all_lang_masks.sum(1)
            lang_token_type_ids = torch.zeros_like(all_lang_masks,
                dtype=torch.long, device=self.device)
            h_t, all_language_features = self.policy.net(
                mode='language',
                lang_idx_tokens=lang_idx_tokens,
                lang_masks=all_lang_masks,
            )
        init_num_envs = self.envs.num_envs

        
        init_bs = len(observations)
        state_not_dones = np.array([True] * init_bs)
        

        il_loss = 0.0
        for stepk in range(self.max_len):
            language_features = all_language_features[not_done_index]       
            lang_masks = all_lang_masks[not_done_index]

            
            if 'VLNBERT' in self.config.MODEL.policy_name:
                language_features = torch.cat(      
                    (h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

            
            positions = []; headings = []
            for ob_i in range(len(observations)):
                agent_state_i = self.envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])

            if 'VLNBERT' in self.config.MODEL.policy_name:
               
                cand_rgb, cand_depth, \
                cand_direction, cand_mask, candidate_lengths, \
                batch_angles, batch_distances = self.policy.net(
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = in_train,
                )
                
                logits, h_t = self.policy.net(      
                    mode = 'navigation',
                    observations=batch,
                    lang_masks=lang_masks,
                    lang_feats=language_features,
                    lang_token_type_ids=lang_token_type_ids,
                    headings=headings,
                    cand_rgb = cand_rgb, 
                    cand_depth = cand_depth,
                    cand_direction = cand_direction,
                    cand_mask = cand_mask,                    
                    masks = not_done_masks,
                )
                

            logits = logits.masked_fill_(cand_mask, -float('inf'))
            total_weight += len(candidate_lengths)

            
            if train_tf:
                cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
                oracle_cand_idx = []
                oracle_stop = []
                for j in range(len(batch_angles)):
                    for k in range(len(batch_angles[j])):
                        angle_k = batch_angles[j][k]
                        forward_k = batch_distances[j][k]
                        dist_k = self.envs.call_at(j, 
                            "cand_dist_to_goal", {
                                "angle": angle_k, "forward": forward_k,
                            })
                        cand_dists_to_goal[j].append(dist_k)
                    curr_dist_to_goal = self.envs.call_at(
                        j, "current_dist_to_goal")
                    
                    if curr_dist_to_goal < 1.5:
                        oracle_cand_idx.append(candidate_lengths[j] - 1)
                        oracle_stop.append(True)
                    else:
                        oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
                        oracle_stop.append(False)

            if train_rl:
                probs = F.softmax(logits, 1)  
                c = torch.distributions.Categorical(probs)
                actions = c.sample().detach()
                rl_entropy = torch.zeros(init_bs, device=self.device)
                rl_entropy[state_not_dones] = c.entropy()
                entropys.append(rl_entropy)
                rl_policy_log_probs = torch.zeros(init_bs, device=self.device)
                rl_policy_log_probs[state_not_dones] = c.log_prob(actions)
                policy_log_probs.append(rl_policy_log_probs)
            elif train_tf:
                oracle_actions = torch.tensor(oracle_cand_idx, device=self.device).unsqueeze(1)
                actions = logits.argmax(dim=-1, keepdim=True)
                actions = torch.where(
                        torch.rand_like(actions, dtype=torch.float) <= self.ratio,
                        oracle_actions, actions)
                current_loss = F.cross_entropy(logits, oracle_actions.squeeze(1), reduction="none")
                ml_loss += torch.sum(current_loss)
            else:
                actions = logits.argmax(dim=-1, keepdim=True)

            
            env_actions = []
           
            for j in range(logits.size(0)):
                if train_rl and (actions[j].item() == candidate_lengths[j]-1 or stepk == self.max_len-1):
                   
                    env_actions.append({'action':
                        {'action': 0, 'action_args':{}}})
                elif actions[j].item() == candidate_lengths[j]-1:
                    
                    env_actions.append({'action':
                        {'action': 0, 'action_args':{}}})
                else:
                   
                    env_actions.append({'action':
                        {'action': 4, 
                        'action_args':{
                            'angle': batch_angles[j][actions[j].item()], 
                            'distance': batch_distances[j][actions[j].item()],
                        }}})

            
            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in
                                             zip(*outputs)]
            
            h_t = h_t[np.array(dones)==False]


            if train_rl:
                rl_actions[state_not_dones] = np.array([sk['action']['action'] for sk in env_actions])

               
                current_dist = np.zeros(init_bs, np.float32)
               
                reward = np.zeros(init_bs, np.float32)
                ct_mask = np.ones(init_bs, np.float32)

                sbi = 0
                for si in range(init_bs):
                    if state_not_dones[si]:
                        info = self.envs.call_at(sbi, "get_metrics", {})
                        current_dist[si] = info['distance_to_goal']
                       
                        sbi += 1

                    if not state_not_dones[si]:
                        reward[si] = 0.0
                        ct_mask[si] = 0.0
                    else:
                        action_idx = rl_actions[si]
                        
                        if action_idx == 0:                            
                            if current_dist[si] < 3.0:                   
                                reward[si] = 2.0 
                            else:                                         
                                reward[si] = -2.0
                        elif action_idx != -100:                                             
                           
                            reward[si] = - (current_dist[si] - last_dist[si])
                            
                            if reward[si] > 0.0:                           
                                reward[si] = 1.0  
                            else:
                                reward[si] = -1.0 
                            
                rewards.append(reward)
                critic_masks.append(ct_mask)
                last_dist[:] = current_dist
                

            state_not_dones[state_not_dones] = np.array(dones) == False

            if sum(dones) > 0:
                shift_index = 0
                for i in range(self.envs.num_envs):
                    if dones[i]:
                        
                        i = i - shift_index
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        if self.envs.num_envs == 0:
                            break

                        observations.pop(i)

                        shift_index += 1

            if self.envs.num_envs == 0:
                break
            not_done_masks = torch.ones(
                self.envs.num_envs, 1, dtype=torch.bool, device=self.device
            )

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )

            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)


        if train_rl:
            rl_loss = 0.
            length = len(rewards)
            discount_reward = np.zeros(init_bs, np.float32)
            rl_total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * 0.90 + rewards[t]  
                mask_ = Variable(torch.from_numpy(critic_masks[t]), requires_grad=False).to(self.device)
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).to(self.device)
                v_ = self.policy.net(
                    mode = 'critic',
                    post_states = hidden_states[t])
                a_ = (r_ - v_).detach()
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  
                rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                rl_total = rl_total + np.sum(critic_masks[t])

            rl_loss = rl_loss / rl_total
            il_loss += rl_loss

        elif train_tf:
            il_loss = ml_loss / total_weight 

        return il_loss  

    def train(self) -> None:
        split = self.config.TASK_CONFIG.DATASET.SPLIT

        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split
        self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = self.config.IL.max_traj_len
        if (
            self.config.IL.DAGGER.expert_policy_sensor
            not in self.config.TASK_CONFIG.TASK.SENSORS
        ):
            self.config.TASK_CONFIG.TASK.SENSORS.append(
                self.config.IL.DAGGER.expert_policy_sensor
            )
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.NUM_ENVIRONMENTS = self.config.IL.batch_size // len(
            self.config.SIMULATOR_GPU_IDS)
        self.config.use_pbar = not is_slurm_batch_job()

        
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

       
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS

       

        self.config.freeze()
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)
            

        self.split = split

        if self.config.RANDOM_ALLOC_SCENE:
            episode_ids = self.allocate_allowed_episode_by_scene_random()
        else:
            episode_ids = self.allocate_allowed_episode_by_scene()


        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME),
            episodes_allowed=episode_ids,
            auto_reset_done=False
        )
        num_epoches_per_ratio = int(np.ceil(self.config.IL.epochs/self.config.IL.decay_time))
        print('\nFinished constructing environments')

        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        

        print('\nInitializing policy network ...')
        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,     
            observation_space=observation_space,
            action_space=action_space,
        )

        

        print('\nTraining starts ...')

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            AuxLosses.activate()
            batches_per_epoch = int(np.ceil(dataset_length/self.batch_size))       
            
            for epoch in range(self.start_epoch, self.config.IL.epochs):
                epoch_str = f"{epoch + 1}/{self.config.IL.epochs}"

                t_ = (
                    tqdm.trange(
                        batches_per_epoch, leave=False, dynamic_ncols=True
                    )
                    if self.config.use_pbar & (self.local_rank < 1)
                    else range(batches_per_epoch)
                )
                self.ratio = np.power(self.config.IL.schedule_ratio, epoch//num_epoches_per_ratio + 1)

                self.trained_episodes = []
                
                if epoch != self.start_epoch:       
                    self.envs = None
                    self.envs = construct_envs(    
                        self.config, get_env_class(self.config.ENV_NAME),
                        episodes_allowed=episode_ids,
                        auto_reset_done=False
                    )
                
                for batch_idx in t_:
                    
                    loss = self.train_ml( 
                        in_train=True, 
                        train_tf=True, train_rl=False)
                    

                    if loss == -1:
                        break
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses = [loss]
                    

                    if self.world_size > 1:
                        for i in range(len(losses)):
                            reduce_loss(losses[i], self.local_rank, self.world_size)
                            losses[i] = losses[i].item()
                    else:
                        for i in range(len(losses)):
                            losses[i] = losses[i].item()
                    loss = losses[0]
                    if self.config.use_pbar:
                        if self.local_rank < 1:  
                            t_.set_postfix(
                                {
                                    "epoch": epoch_str,
                                    "loss": round(loss, 4),
                                }
                            )

                            writer.add_scalar("loss", loss, self.step_id)
                    self.step_id += 1  

                if self.local_rank < 1 and epoch % self.config.save_interval == self.config.save_plus:
                    self.save_checkpoint(epoch, self.step_id)

                AuxLosses.deactivate()
