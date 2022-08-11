import json
import numpy as np


def merge_captions_question_programs(path_cap, path_ques, caption_first=True):
    with open(path_cap, "r"):
        c_progs = path_cap.readlines()
    with open(path_ques, "r"):
        q_progs = path_ques.readlines()

    all_merged_progs = []
    i = 0
    while i < len(q_progs):
        cap_idx = i % 11 if caption_first else i % 10
        start_idx_p = i + 1 if caption_first else i
        end_idx_p = start_idx_p + 12 if caption_first else  start_idx_p + 11
        temp = c_progs[cap_idx] + q_progs[start_idx_p, end_idx_p]
        all_merged_progs.append(temp)
        i = end_idx_p


def load_clevr_scenes(scenes_json):
    with open(scenes_json) as f:
        scenes_raw = json.load(f)
    if type(scenes_raw) == dict:
        scenes_raw = scenes_raw["scenes"]

    scenes = []
    for s in scenes_raw:
        table = []
        for i, o in enumerate(s['objects']):
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)
            if '3d_coords' in o:
                item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                                    np.dot(o['3d_coords'], s['directions']['front']),
                                    o['3d_coords'][2]]
            else:
                item['position'] = o['position']
            item['color'] = o['color']
            item['material'] = o['material']
            item['shape'] = o['shape']
            item['size'] = o['size']
            table.append(item)
        scenes.append(table)
    return scenes


def load_minecraft_scenes(scenes_json):
    with open(scenes_json) as f:
        scenes_raw = json.load(f)
    if type(scenes_raw) == dict:
        scenes_raw = scenes_raw["scenes"]

    scenes = []
    for s in scenes_raw:
        table = []
        for i, o in enumerate(s['objects']):
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)
            if '3d_coords' in o:
                item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                                    np.dot(o['3d_coords'], s['directions']['front']),
                                    o['3d_coords'][2]]
            else:
                item['position'] = o['position']
            item['nature'] = o['nature']
            item['class'] = o['class']
            item['direction'] = "facing_"
            if o['direction'] == "front":
                item['direction'] += "forward"
            elif o['direction'] == "back":
                item['direction'] += "backward"
            elif o['direction'] == "right":
                item['direction'] += "right"
            elif o['direction'] == "left":
                item['direction'] += "left"
            table.append(item)
        scenes.append(table)
    return scenes
