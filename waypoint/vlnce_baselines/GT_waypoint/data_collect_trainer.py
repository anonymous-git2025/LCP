import json
import jsonlines
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)

from vlnce_baselines.ss_trainer_VLNBERT import SSTrainer

from habitat.utils.geometry_utils import quaternion_from_coeff

@baseline_registry.register_trainer(name="data_collect")
class Data_Collect_Trainer(SSTrainer):
    def __init__(self, config=None):
        super().__init__(config)

    
    def get_pos_ori(self, envs):
        pos_ori = envs.call(['get_pos_ori']*envs.num_envs)        
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori
    
    def get_pos_ori_idx(self, envs, idx):
        pos_ori = envs.call(['get_pos_ori']*envs.num_envs) 
        pos = pos_ori[idx][0]
        ori = pos_ori[idx][1]
        return pos, ori
    
    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        prev_node_idx,
        current_node_idx,
        waypoint_predictors,
        batch,
        rgb_frames=None,
    ):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                
                prev_node_idx.pop(idx)
                current_node_idx.pop(idx)
                waypoint_predictors.pop(idx)
                

            
            recurrent_hidden_states = recurrent_hidden_states[state_index]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            recurrent_hidden_states,
            not_done_masks,
            prev_actions,
            prev_node_idx,
            current_node_idx,
            waypoint_predictors,
            batch,
            rgb_frames,
        )
    
    
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()
        config.defrost()
        
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                print("skipping -- evaluation exists.")
                return

        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj
        )

        dataset_length = sum(envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(config)
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval()
        

        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.GT_waypoint.gt_waypoint import GT_waypoint
        self.waypoint_predictors = [GT_waypoint(config=self.config.DATA_COLLECT) for _ in range(envs.num_envs)]
        
        scene_name = [episode.scene_id.split('/')[-1][:-4] for episode in envs.current_episodes()]
        first_pos, first_ori = self.get_pos_ori(envs)
        current_node_idx = []
        observations = []
        for idx in range(envs.num_envs):
            first_node_idx, first_node_position, _ = self.waypoint_predictors[idx].init_GT_waypoint(scene_name=scene_name[idx], first_node_pos=first_pos[idx], first_node_ori=first_ori[idx])
            
            current_node_idx.append(first_node_idx)
            envs.call_at(idx, "teleport_with_quat", {"pos": first_node_position, "quat": quaternion_from_coeff(first_ori[idx])})
            observations.append(envs.call_at(idx, "get_observation_at", {"source_position": first_node_position,"source_rotation": quaternion_from_coeff(first_ori[idx])}))
        
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        prev_node_idx = [None] * envs.num_envs
        
        """
        predictor
        """


        if 'CMA' in self.config.MODEL.policy_name:
            rnn_states = torch.zeros(
                envs.num_envs,
                self.num_recurrent_layers,
                config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            )
        elif 'VLNBERT' in self.config.MODEL.policy_name:
            h_t = torch.zeros(
                envs.num_envs, 768,
                device=self.device,
            )
            language_features = torch.zeros(
                envs.num_envs, 80, 768,
                 device=self.device,
            )
       
        
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        total_weight = 0.
        ml_loss = 0.
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()
            positions = []; headings = []
            for ob_i in range(len(current_episodes)):
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])

            with torch.no_grad():
                if 'CMA' in self.config.MODEL.policy_name:
                   
                    instruction_embedding, all_lang_masks = self.policy.net(
                        mode = "language",
                        observations = batch,
                    )

                    
                    cand_rgb, cand_depth, \
                    cand_direction, cand_mask, candidate_lengths, \
                    batch_angles, batch_distances = self.policy.net(
                        mode = "waypoint",
                        waypoint_predictor = self.waypoint_predictor,
                        observations = batch,
                        in_train = False,
                    )
                    
                    logits, rnn_states = self.policy.net(
                        mode = 'navigation',
                        observations = batch,
                        instruction = instruction_embedding,
                        text_mask = all_lang_masks,
                        rnn_states = rnn_states,
                        headings = headings,
                        cand_rgb = cand_rgb, 
                        cand_depth = cand_depth,
                        cand_direction = cand_direction,
                        cand_mask = cand_mask,
                        masks = not_done_masks,
                    )
                    logits = logits.masked_fill_(cand_mask, -float('inf'))

                elif 'VLNBERT' in self.config.MODEL.policy_name:
                  
                    lang_idx_tokens = batch['instruction']
                    padding_idx = 0
                    lang_masks = (lang_idx_tokens != padding_idx)
                    lang_token_type_ids = torch.zeros_like(lang_masks,
                        dtype=torch.long, device=self.device)
                    h_t_flag = h_t.sum(1)==0.0      
                    h_t_init, language_features = self.policy.net(     
                        mode='language',
                        lang_idx_tokens=lang_idx_tokens,
                        lang_masks=lang_masks)
                    h_t[h_t_flag] = h_t_init[h_t_flag]      
                    language_features = torch.cat(      
                        (h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)
                    
                    
                    
                  
                    source_info = [
                        {'current_node_idx': current_node_idx[i], 
                        'prev_node_idx': prev_node_idx[i]}
                        for i in range(envs.num_envs)
                    ]


                    cand_rgb, cand_depth, \
                    cand_direction, cand_mask, candidate_lengths, \
                    batch_angles, batch_distances, \
                    batch_idx, batch_pos = self.policy.net(
                        mode = "GT_waypoint",
                        waypoint_predictor = self.waypoint_predictors,
                        observations = batch,
                        in_train = False,
                        source_info = source_info
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

                
                actions = logits.argmax(dim=-1, keepdim=True)
                env_actions = []
                for j in range(logits.size(0)):
                    if actions[j].item() == candidate_lengths[j]-1:
                        env_actions.append({'action':
                            {'action': 0, 'action_args':{}}})
                    else:
                        env_actions.append({'action':
                            {'action': 4, 
                            'action_args':{
                                'angle': batch_angles[j][actions[j].item()], 
                                'distance': batch_distances[j][actions[j].item()],
                                'pos': batch_pos[j][actions[j].item()]
                            }}})
                        
                        prev_node_idx[j] = current_node_idx[j]
                        current_node_idx[j] = batch_idx[j][actions[j].item()]

            outputs = envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            for j, ob in enumerate(observations):
                if env_actions[j]['action']['action'] == 0:
                    continue
                else:       
                    envs.call_at(j, 
                        'change_current_path',
                        {'new_path': ob.pop('positions'),
                        'collisions': ob.pop('collisions')}
                    )

            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8, device=self.device)

            
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)

                if not dones[i]:        
                    continue

                info = infos[i]
                metric = {}
                metric['steps_taken'] = info['steps_taken']
                ep_id = str(envs.current_episodes()[i].episode_id)
                gt_path = np.array(self.gt_data[ep_id]['locations']).astype(np.float)
                if 'current_path' in envs.current_episodes()[i].info.keys():
                    positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(np.float)
                    collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                    assert collisions_.shape[0] == positions_.shape[0] - 1
                else:
                    positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(np.float)
                distance = np.array(info['position']['distance']).astype(np.float)
                metric['distance_to_goal'] = distance[-1]
                metric['success'] = 1. if distance[-1] <= 3. and env_actions[i]['action']['action'] == 0 else 0.
                metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                metric['collisions'] = collisions_.mean()
                gt_length = distance[0]
                metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])

                act_con_path = positions_
                gt_con_path = np.array(gt_path).astype(np.float)
                dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                metric['ndtw'] = nDTW
                stats_episodes[current_episodes[i].episode_id] = metric

                observations[i] = envs.reset_at(i)[0]       


                scene_name = envs.current_episodes()[i].scene_id.split('/')[-1][:-4]
                first_pos, first_ori = self.get_pos_ori_idx(envs, i)
                first_node_idx, first_node_position, _ = self.waypoint_predictors[i].init_GT_waypoint(scene_name=scene_name, first_node_pos=first_pos, first_node_ori=first_ori)
                
                current_node_idx[i] = first_node_idx
                envs.call_at(i, "teleport_with_quat", {"pos": first_node_position, "quat": quaternion_from_coeff(first_ori)})
                observations[i] = envs.call_at(i, "get_observation_at", {"source_position": first_node_position,"source_rotation": quaternion_from_coeff(first_ori)})

                prev_node_idx[i] = None



                if 'CMA' in self.config.MODEL.policy_name:
                    rnn_states[i] *= 0.
                elif 'VLNBERT' in self.config.MODEL.policy_name:
                    h_t[i] *= 0.        

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=episodes_to_eval,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            "spl": stats_episodes[
                                current_episodes[i].episode_id
                            ]["spl"]
                        },
                        tb_writer=writer,
                    )

                    
                    del stats_episodes[current_episodes[i].episode_id][
                        "collisions"
                    ]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:       
                    envs_to_pause.append(i)

            if 'VLNBERT' in self.config.MODEL.policy_name:
                rnn_states = h_t

            headings = torch.tensor(headings)
            (
                envs,
                rnn_states,
                not_done_masks,
                headings,  
                prev_node_idx,
                current_node_idx,
                self.waypoint_predictors,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                rnn_states,
                not_done_masks,
                headings,
                prev_node_idx,
                current_node_idx,
                self.waypoint_predictors,       
                batch,
                rgb_frames
            )
            headings = headings.tolist()
            if 'VLNBERT' in self.config.MODEL.policy_name:
                h_t = rnn_states

        envs.close()
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            dist.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_stats[k] = v

        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
       

        self.policy.to(self.device)
        
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=True, broadcast_buffers=False)
            

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.config.IL.lr,
        )

        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"])
                self.policy.net = self.policy.net.module
                
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"])
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    

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
        
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.GT_waypoint.gt_waypoint import GT_waypoint
        self.waypoint_predictors = [GT_waypoint(config=self.config.DATA_COLLECT) for _ in range(self.envs.num_envs)]
        
        scene_name = [episode.scene_id.split('/')[-1][:-4] for episode in self.envs.current_episodes()]
        first_pos, first_ori = self.get_pos_ori(self.envs)
        current_node_idx = []
        observations = []
        for idx in range(self.envs.num_envs):
            first_node_idx, first_node_position, _ = self.waypoint_predictors[idx].init_GT_waypoint(scene_name=scene_name[idx], first_node_pos=first_pos[idx], first_node_ori=first_ori[idx])
            
            current_node_idx.append(first_node_idx)
            self.envs.call_at(idx, "teleport_with_quat", {"pos": first_node_position, "quat": quaternion_from_coeff(first_ori[idx])})
            observations.append(self.envs.call_at(idx, "get_observation_at", {"source_position": first_node_position,"source_rotation": quaternion_from_coeff(first_ori[idx])}))
        
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        prev_node_idx = [None] * self.envs.num_envs
        
        """
        predictor
        """
        

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
               
                source_info = [
                    {'current_node_idx': current_node_idx[i], 
                    'prev_node_idx': prev_node_idx[i]}
                    for i in range(self.envs.num_envs)
                ]


                cand_rgb, cand_depth, \
                cand_direction, cand_mask, candidate_lengths, \
                batch_angles, batch_distances, \
                batch_idx, batch_pos = self.policy.net(
                    mode = "GT_waypoint",
                    waypoint_predictor = self.waypoint_predictors,
                    observations = batch,
                    in_train = False,
                    source_info = source_info
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
                            'pos': batch_pos[j][actions[j].item()]
                        }}})
                    
                    prev_node_idx[j] = current_node_idx[j]
                    current_node_idx[j] = batch_idx[j][actions[j].item()]

            
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
                        
                        prev_node_idx.pop(i)
                        current_node_idx.pop(i)
                        self.waypoint_predictors.pop(i)
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