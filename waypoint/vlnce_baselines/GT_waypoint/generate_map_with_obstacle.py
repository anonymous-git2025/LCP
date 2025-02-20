import json
import numpy as np
import utils
import habitat
from habitat.sims import make_sim
import os

connect_with_nav_path = 'vlnce_baselines/GT_waypoint/connect_with_nav/%s/%s.json'
scene_path = 'waypoint_predict/waypoint-predictor/data/scene_datasets/mp3d/{scan}/{scan}.glb'
RAW_GRAPH_PATH= 'waypoint_predict/waypoint-predictor/habitat_connectivity_graph/official_11_21/%s.json'
config_path = 'vlnce_baselines/GT_waypoint/config.yaml'
log_path = 'vlnce_baselines/GT_waypoint/log_3.txt'

splits = ['train', 'val_unseen']

for split in splits:
    with open(RAW_GRAPH_PATH%split, 'r') as f:
        raw_graph_data = json.load(f)
    
    del_edge = 0
    del_node = 0
    total_edge = 0
    total_node = 0
    node_with_edge = 0
    edge_num = 0
    node_num = 0
    error_node = {}
    
    for scene, data in raw_graph_data.items():
        ''' connectivity dictionary '''
        connect_dict = {}
        del_edge_scene = 0
        del_node_scene = 0
        error_node_scene = []
        total_edge_scene = len(data['edges']) * 2   
        total_node_scene = len(data['nodes'])      
        total_edge += total_edge_scene
        total_node += total_node_scene
        
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

        node_with_edge_scene = len(connect_dict.keys())
        node_with_edge += node_with_edge_scene

        '''make sim for obstacle checking'''
        config = habitat.get_config(config_path)
        config.defrost()
       
        config.SIMULATOR.FORWARD_STEP_SIZE = 0.01
        config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
        config.SIMULATOR.TYPE = 'Sim-v1'
        config.SIMULATOR.SCENE = scene_path.format(scan=scene)
        sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

        ''' process each node to standard data format '''
        navigability_connect_dict = {}
        total = len(connect_dict)
        for i, pair in enumerate(connect_dict.items()):
            node_a, neighbors = pair        
            navigability_connect_dict[node_a] = []      
            node_a_pos = np.array(data['nodes'][node_a])[[0,2]]     
        
            habitat_pos = np.array(data['nodes'][node_a])
        

            for node_b in neighbors:
                
                node_b_pos = np.array(data['nodes'][node_b])[[0,2]]
        
                edge_vec = (node_b_pos - node_a_pos)
                angle, distance = utils.edge_vec_to_ang_dis(edge_vec)      

                
                if utils.navigable(habitat_pos, angle, distance, sim, config.SIMULATOR.FORWARD_STEP_SIZE):
                    navigability_connect_dict[node_a].append(node_b)
                else:
                    error_node_scene.append((node_a, node_b))
                    del_edge_scene += 1
        

            utils.print_progress(i+1,total)

        for pair in error_node_scene:
            if pair[0] in navigability_connect_dict[pair[1]]:
                navigability_connect_dict[pair[1]].remove(pair[0])
                del_edge_scene += 1
                error_node_scene.append((pair[1], pair[0]))

        edge_num_scene = 0
        node_num_scene = 0      
        for node_a in navigability_connect_dict.keys():
            edge_num_scene += len(navigability_connect_dict[node_a])
            if len(navigability_connect_dict[node_a]) == 0:
                del(navigability_connect_dict[node_a])
                del_node_scene += 1
            else:
                node_num_scene += 1
        
        
        del_edge_rate_scene = del_edge_scene / total_edge_scene
        del_node_rate_scene = del_node_scene / node_with_edge_scene
        del_edge += del_edge_scene
        del_node += del_node_scene
        edge_num += edge_num_scene      
        node_num += node_num_scene
        
        if len(error_node_scene) != 0:
            error_node[scene] = error_node_scene

        with open(connect_with_nav_path%(split,scene), 'w') as fo:
            json.dump(navigability_connect_dict, fo, ensure_ascii=False, indent=4)


        
        print('Total / With edge / Count / Delete number of nodes in %s: %s / %s / %s / %s / %s'%(scene, total_node_scene, node_with_edge_scene, node_num_scene, del_node_scene, del_node_rate_scene) )
        print('Total / Count / Delete number of edges in %s: %s / %s / %s / %s'%(scene, total_edge_scene, edge_num_scene, del_edge_scene, del_edge_rate_scene) )
        
        
        with open(log_path, "a") as f:
            f.write('Total / With edge / Count / Delete number of nodes in %s: %s / %s / %s / %s / %s'%(scene, total_node_scene, node_with_edge_scene, node_num_scene, del_node_scene, del_node_rate_scene))
            f.write('\n')
            f.write('Total / Count / Delete number of edges in %s: %s / %s / %s / %s'%(scene, total_edge_scene, edge_num_scene,  del_edge_scene, del_edge_rate_scene))
            f.write('\n')
            if len(error_node_scene) != 0:
                f.write(json.dumps(error_node,ensure_ascii=False, indent=4))
                f.write('\n')
            
        
        sim.close()

    del_edge_rate = del_edge / total_edge
    del_node_rate = del_node / node_with_edge

    print('Total / With edge / Count / Delete number of nodes in %s set: %s / %s / %s / %s / %s'%(split, total_node, node_with_edge, node_num, del_node, del_node_rate))
    print('Total / Count / Delete number of edges in %s set: %s / %s / %s / %s'%(split, total_edge, edge_num, del_edge, del_edge_rate))
    with open(log_path, "a") as f:
        f.write('=======================================\n')
        f.write('Total / With edge / Count / Delete number of nodes in %s set: %s / %s / %s / %s / %s'%(split, total_node, node_with_edge, node_num, del_node, del_node_rate))
        f.write('\n')
        f.write('Total / Count / Delete number of edges in %s set: %s / %s / %s / %s'%(split, total_edge, edge_num, del_edge, del_edge_rate))
        f.write('\n')
        f.write('=======================================\n')