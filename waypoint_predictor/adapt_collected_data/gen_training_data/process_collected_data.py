import pickle as pkl
import os

data_path = './data_collect'


splits = ['val_seen', 'val_unseen', 'train']

jump_scene = pkl.load(open('data_collect/process/jump_scene.pkl', 'rb'))

for split in splits:
    split_path = data_path + '/' + split
    for root, dirs, files in os.walk(split_path):
        for scene in dirs:
            if split == 'val_seen' and scene in jump_scene:
                continue
            scene_path = split_path + '/' + scene
            for root, dirs, files in os.walk(scene_path):
                for file in files:
                    episode = pkl.load(open(scene_path + '/' + file, 'rb'))
                    for i, step in enumerate(episode['step']):
                        if i == 0:
                            step['prev_lang'] = None
                        else:
                            step['prev_lang'] = {
                                'h_t': episode['step'][i - 1]['h_t'],
                                'lang_state_score': episode['step'][i - 1]['lang_state_scores'],
                                'attened_lang': episode['step'][i - 1]['attened_lang']
                            }
                        step['step_id'] = i
                        step['episode_info'] = episode['info']
                        with open(scene_path + '/' + scene + '_' + file[:-4] + '_' + str(i) + '_' + step['node_idx'] + '.pkl', 'wb') as f:
                             pkl.dump(step, f)

                    os.remove(scene_path + '/' + file)
                    
            print(scene, 'done!')
    print('Split done!')
