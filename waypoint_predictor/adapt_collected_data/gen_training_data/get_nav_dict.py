import json
import numpy as np
import utils
import habitat
from habitat.sims import make_sim
from utils import Simulator
import os
import pickle as pkl
import copy

config_path = 'gen_training_data/config.yaml'
scene_path = 'data/scene_datasets/mp3d/{scan}/{scan}.glb'
RAW_GRAPH_PATH= 'habitat_connectivity_graph/official/%s.json'
data_path = 'data_collect/{split}/{scene_name}'
NUMBER = 120

splits = ['val_seen', 'val_unseen', 'train']

for SPLIT in splits:

    with open(RAW_GRAPH_PATH%SPLIT, 'r') as f:
        raw_graph_data = json.load(f)

    nav_dict = {}
    total_invalids = 0
    total = 0

    for scene, data in raw_graph_data.items():      
        ''' connectivity dictionary '''
        connect_dict = {}
        for edge_id, edge_info in data['edges'].items():
            node_a = edge_info['nodes'][0]
            node_b = edge_info['nodes'][1]

            if node_a not in connect_dict:
                connect_dict[node_a] = [node_b]
            else:
                connect_dict[node_a].append(node_b)
            if node_b not in connect_dict:
                connect_dict[node_b] = [node_a]
            else:
                connect_dict[node_b].append(node_a)


        '''make sim for obstacle checking'''
        config = habitat.get_config(config_path)
        config.defrost()
        
        config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
        config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
        config.SIMULATOR.TYPE = 'Sim-v1'
        config.SIMULATOR.SCENE = scene_path.format(scan=scene)
        sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)


        
        ''' process each node to standard data format '''
        episodes_info = {}
        data_dir = data_path.format(split=SPLIT, scene_name=scene)
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                episode_id = file[:-4]
                episode = pkl.load(open(data_dir + '/' + file,"rb"))
                node_info = [{
                    'heading': node['heading'],
                    'node_id': node['node_idx']} 
                    for node in episode['step']
                ]
                episodes_info[episode_id] = copy.deepcopy(node_info)

        episode_nav_dict = {}
        total = len(episodes_info)
        for i, (episode_id, steps) in enumerate(episodes_info.items()):
            navigability_dict_list = []
            for node_info in steps:
                navigability_dict = {}
                node_id = node_info['node_id']
                neighbors = connect_dict[node_id]
                navigability_dict[node_id] = utils.init_single_node_dict(number=NUMBER)     
                node_a_pos = np.array(data['nodes'][node_id])[[0,2]]   
                
                habitat_pos = np.array(data['nodes'][node_id])
                for id, info in navigability_dict[node_id].items():      
                    
                    obstacle_distance, obstacle_index = utils.get_obstacle_info(habitat_pos, info['heading'], node_info['heading'], sim)       
                    info['obstacle_distance'] = obstacle_distance       
                    info['obstacle_index'] = obstacle_index    

                for node_b in neighbors:
                    node_b_pos = np.array(data['nodes'][node_b])[[0,2]]
            
                    edge_vec = (node_b_pos - node_a_pos)       
                    angle, angleIndex, distance, distanceIndex = utils.edge_vec_to_indexes(edge_vec,node_info['heading'],number=NUMBER)     
            
                    navigability_dict[node_id][str(angleIndex)]['has_waypoint'] = True
                    navigability_dict[node_id][str(angleIndex)]['waypoint'].append(
                        {
                            'node_id': node_b,
                            'position': node_b_pos.tolist(),
                            'angle': angle,
                            'angleIndex': angleIndex,
                            'distance': distance,
                            'distanceIndex': distanceIndex,
                        })         
                navigability_dict_list.append(navigability_dict)
            episode_nav_dict[episode_id] = navigability_dict_list
            utils.print_progress(i+1,total)
        nav_dict[scene] = episode_nav_dict
        sim.close()

    output_dir = './adapt_collected_data/gen_training_data/nav_dicts'
    output_path = output_dir + '/navigability_dict_%s.json'%SPLIT
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, 'w') as fo:
        json.dump(nav_dict, fo, ensure_ascii=False, indent=4)
