import json
import numpy as np
import vlnce_baselines.GT_waypoint.utils as utils



class GT_waypoint(object):
    def __init__(self, config=None):
        self.connectivity_graph_path = config.connectivity_graph_path
        self.connect_dict_path = config.connect_dict_path
        self.loc_noise = config.loc_noise
        self.connectivity_graph = {}
        self.connect_dict = {}
        
    
    def init_GT_waypoint(self, scene_name=None, first_node_pos=None, first_node_ori=None):
        self.scene_name = scene_name
        with open(self.connectivity_graph_path + '/' + scene_name + '.json', 'r') as f:
            self.connectivity_graph = json.load(f)

        self.scene_name = scene_name
        
        first_node_idx = self._identify_by_loc(first_node_pos, self.connectivity_graph['nodes'])
        
        
        with open(self.connect_dict_path + '/' + scene_name + '.json', 'r') as f:
            self.connect_dict = json.load(f)
        
        self.first_node_heading = utils.heading_from_quaternion(first_node_ori)
        return first_node_idx, self.connectivity_graph['nodes'][first_node_idx], self.first_node_heading
    
    def _identify_by_loc(self, qpos, kpos_dict):
        min_dis = 10000
        min_vp = None
        for kvp, kpos in kpos_dict.items():     
            dis = ((qpos - kpos)**2).sum()**0.5     
            if dis < min_dis:       
                min_dis = dis
                min_vp = kvp
        if min_dis > self.loc_noise:
            raise ValueError('NoTargetNode')
        return min_vp



    def get_cand(self, source_info):
        

        current_node_idx = source_info['current_node_idx']     
        
        source_pos = self.connectivity_graph['nodes'][current_node_idx]
        if source_info['prev_node_idx']:
            prev_pos = self.connectivity_graph['nodes'][source_info['prev_node_idx']]
            source_heading, _ = utils.calculate_vp_rel_pos(p1=prev_pos, p2=source_pos)
        else:
            source_heading = self.first_node_heading

        cands = self.connect_dict[current_node_idx]
        cand_info = {'idx': [], 'ang': [], 'dis': [], 'pos': []}
        for cand in cands:
            cand_pos = np.array(self.connectivity_graph['nodes'][cand])
            ang, dis = utils.calculate_vp_rel_pos(p1=source_pos, p2=cand_pos, base_heading=source_heading)
            
            cand_info['idx'].append(cand)
            cand_info['ang'].append(ang)
            cand_info['dis'].append(dis)
            cand_info['pos'].append(cand_pos)
        return cand_info

