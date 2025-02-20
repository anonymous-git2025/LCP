import json
import math
import numpy as np
import copy
import torch
import os
import utils
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

import pickle as pkl

ANGLES = 120
DISTANCES = 12
RAW_GRAPH_PATH = './habitat_connectivity_graph/official/%s.json'
data_path = 'data_collect/{split}/{scene_name}'

RADIUS = 3.25  

print('Running TRM-0.2 !!!!!!!!!!')

print('\nProcessing navigability and connectivity to GT maps')
print('Using %s ANGLES and %s DISTANCES'%(ANGLES, DISTANCES))
print('Maximum radius for each waypoint: %s'%(RADIUS))
print('\nConstraining each angle sector has at most one GT waypoint')
print('For all sectors with edges greater than %s, create a virtual waypoint at %s'%(RADIUS, 2.50))
print('\nThis script will return the target map, the obstacle map and the weigh map for each environment')

np.random.seed(7)

splits = ['val_unseen', 'train', 'val_seen']
for split in splits:
    print('\nProcessing %s split data'%(split))

    with open(RAW_GRAPH_PATH%split, 'r') as f:
        data = json.load(f)
    if os.path.exists('./adapt_collected_data/gen_training_data/nav_dicts/navigability_dict_%s.json'%split):
        with open('./adapt_collected_data/gen_training_data/nav_dicts/navigability_dict_%s.json'%split) as f:
            nav_dict = json.load(f)
    raw_nav_dict = {}
    nodes = {}
    edges = {}
    obstacles = {}
    for k, v in data.items():
        nodes[k] = data[k]['nodes']
        edges[k] = data[k]['edges']
        obstacles[k] = nav_dict[k]
    raw_nav_dict['nodes'], raw_nav_dict['edges'], raw_nav_dict['obstacles'] = nodes, edges, obstacles
    data_scans = {
        'nodes': raw_nav_dict['nodes'],
        'edges': raw_nav_dict['edges'],
    }
    obstacle_dict_scans = raw_nav_dict['obstacles']     
    scans = list(raw_nav_dict['nodes'].keys())


    overall_nav_dict = {}
    del_nodes = 0
    count_nodes = 0
    target_count = 0 
    openspace_count = 0; obstacle_count = 0
    rawedges_count = 0; postedges_count = 0

    del_node_idx = set([])

    for scan in scans:
        ''' connectivity dictionary '''
        obstacle_dict = obstacle_dict_scans[scan]

        connect_dict = {}
        for edge_id, edge_info in data_scans['edges'][scan].items():
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

        
        episodes_info = {}
        data_dir = data_path.format(split=split, scene_name=scan)
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


        ''' process each node to standard data format '''
        count_nodes_i = 0
        del_nodes_i = 0
        scan_episode = {}
        for i, (episode_id, steps) in enumerate(episodes_info.items()):
            groundtruth_dict_list = []
            for step_id, node_info in enumerate(steps):
                navigability_dict = {}
                groundtruth_dict = {}
                node_id = node_info['node_id']
                neighbors = connect_dict[node_id]
                count_nodes += 1; count_nodes_i += 1
                navigability_dict[node_id] = utils.init_node_nav_dict(ANGLES)
                groundtruth_dict[node_id] = utils.init_node_gt_dict(ANGLES)

                node_a_pos = np.array(data_scans['nodes'][scan][node_id])[[0,2]]
                groundtruth_dict[node_id]['source_pos'] = node_a_pos.tolist()

                for node_b in neighbors:
                    node_b_pos = np.array(data_scans['nodes'][scan][node_b])[[0,2]]   

                    edge_vec = (node_b_pos - node_a_pos)
                    angle, angleIndex, \
                    distance, \
                    distanceIndex = utils.edge_vec_to_indexes(
                        edge_vec, node_info['heading'], ANGLES)   

                    
                    if distanceIndex == -1:     
                        continue
                    
                    if navigability_dict[node_id][str(angleIndex)]['has_waypoint']:     
                        if distance < navigability_dict[node_id][str(angleIndex)]['waypoint']['distance']:       
                            continue

                    navigability_dict[node_id][str(angleIndex)]['has_waypoint'] = True
                    navigability_dict[node_id][str(angleIndex)]['waypoint'] = {     
                            'node_id': node_b,
                            'position': node_b_pos,
                            'angle': angle,
                            'angleIndex': angleIndex,
                            'distance': distance,
                            'distanceIndex': distanceIndex,
                        }
                    ''' set target map '''
                    groundtruth_dict[node_id]['target'][angleIndex, distanceIndex] = 1      
                    groundtruth_dict[node_id]['target_pos'].append(node_b_pos.tolist())      

               
                raw_target_count = groundtruth_dict[node_id]['target'].sum()    

                if raw_target_count == 0:      
                    del(groundtruth_dict[node_id])      
                    groundtruth_dict_list.append(None)     
                    del_nodes += 1; del_nodes_i += 1   
                    del_node_idx.add(node_id)             
                    continue

                ''' a Gaussian target map '''
                gau_peak = 10
                gau_sig_angle = 1.0
                gau_sig_dist = 2.0
                groundtruth_dict[node_id]['target'] = groundtruth_dict[node_id]['target'].astype(np.float32)

                gau_temp_in = np.concatenate(
                    (
                        np.zeros((ANGLES,10)),
                        groundtruth_dict[node_id]['target'],
                        np.zeros((ANGLES,10)),
                    ), axis=1)      

                gau_target = gaussian_filter(
                    gau_temp_in,
                    sigma=(1,2),
                    mode='wrap',       
                )     
                gau_target = gau_target[:, 10:10+DISTANCES]    

                gau_target_maxnorm = gau_target / gau_target.max()
                groundtruth_dict[node_id]['target'] = gau_peak * gau_target_maxnorm    

                for k in range(ANGLES):
                    k_dist = obstacle_dict[episode_id][step_id][node_id][str(k)]['obstacle_distance']     
                    if k_dist is None:
                        k_dist = 100
                    navigability_dict[node_id][str(k)]['obstacle_distance'] = k_dist    

                    k_dindex = utils.get_obstacle_distanceIndex12(k_dist)      
                    navigability_dict[node_id][str(k)]['obstacle_index'] = k_dindex

                    ''' deal with obstacle '''
                    if k_dindex != -1:
                        groundtruth_dict[node_id]['obstacle'][k][:k_dindex] = np.zeros(k_dindex)    
                    else:
                        groundtruth_dict[node_id]['obstacle'][k] = np.zeros(12)     



                rawt = copy.deepcopy(groundtruth_dict[node_id]['target'])       

                groundtruth_dict[node_id]['target'] = \
                    groundtruth_dict[node_id]['target'] * (groundtruth_dict[node_id]['obstacle'] == 0)      

              
                if groundtruth_dict[node_id]['target'].max() < 0.75*gau_peak:      
                    del(groundtruth_dict[node_id])
                    groundtruth_dict_list.append(None)
                    del_nodes += 1; del_nodes_i += 1
                    del_node_idx.add(node_id) 
                    continue

                postt = copy.deepcopy(groundtruth_dict[node_id]['target'])      
                rawedges_count += (rawt==gau_peak).sum()       
                postedges_count += (postt==gau_peak).sum()      

               

                openspace_count += (groundtruth_dict[node_id]['obstacle'] == 0).sum()       
                obstacle_count += (groundtruth_dict[node_id]['obstacle'] == 1).sum()

                groundtruth_dict[node_id]['target'] = groundtruth_dict[node_id]['target'].tolist()
                groundtruth_dict[node_id]['weight'] = groundtruth_dict[node_id]['weight'].tolist()
                groundtruth_dict[node_id]['obstacle'] = groundtruth_dict[node_id]['obstacle'].tolist()

                groundtruth_dict_list.append(groundtruth_dict)

            scan_episode[episode_id] = groundtruth_dict_list
        overall_nav_dict[scan] = scan_episode


    print('Obstacle comes before target !!!')
    print('Number of deleted nodes: %s/%s'%(del_nodes, count_nodes))     
    print('Ratio of obstacle behind target: %s/%s'%(postedges_count,rawedges_count))       

    print('Ratio of openspace %.5f'%(openspace_count/(openspace_count+obstacle_count)))
    print('Ratio of obstacle %.5f'%(obstacle_count/(openspace_count+obstacle_count)))

    print('Deleted nodes:', del_node_idx)

    with open('./adapt_collected_data/training_data/%s_%s_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'%(ANGLES, split), 'w') as f:
        json.dump(overall_nav_dict, f)
    print('Done')


