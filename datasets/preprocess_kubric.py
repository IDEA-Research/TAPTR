import os
import numpy as np
import json
from tqdm import tqdm

#! Collect the sparse annotations of Kubric dataset into a single file for better reading speed in dataset.

def collect_annotations(dir_kubric, image_set):
    """Collect all the annotations of Kubric dataset into a single file for better reading speed in dataset.
    This function will save the annotations at dir_kubric/annotations/{image_set}.json.

    Args:
        dir_kubric (str): the root directory of kubric dataset. 
        image_set (_type_): the split of the dataset, e.g. 'train', 'val', 'test'.
    """
    def sort_video_ids(video_ids):
        """sort the video ids in ascending order. where the video ids are numbers in strings format.

        Args:
            video_ids (List[str]): the video ids

        Returns:
            sorted_video_ids (List[str]): the sorted video ids in string format.
        """
        video_ids_int = [int(video_id) for video_id in video_ids]
        sorted_video_ids = [video_id for _, video_id in sorted(zip(video_ids_int, video_ids))]
        return sorted_video_ids
    dir_data = os.path.join(dir_kubric, image_set)
    video_ids = os.listdir(dir_data)
    video_ids = sort_video_ids(video_ids)
    annots_all = {}
    for video_id in tqdm(video_ids):
        path_anno = os.path.join(dir_data, video_id, f'{video_id}.npy')
        annots = np.load(path_anno, allow_pickle=True).item()
        num_frame = len(os.listdir(os.path.join(dir_data, video_id, 'frames')))
        annots = {
            "points": annots["coords"].tolist(),
            "occluded": annots["visibility"].tolist(),
            "num_frames": num_frame
        }
        annots_all[video_id] = annots
    os.makedirs(os.path.join(dir_kubric, 'annotations'), exist_ok=True)
    path_save = os.path.join(dir_kubric, f'annotations/{image_set}.json')
    print("Saving to", path_save)
    with open(path_save, 'w') as f:
        json.dump(annots_all, f)

    

if __name__ == '__main__':
    dir_kubric = 'datas/kubric_movif_cotracker/r256'
    collect_annotations(dir_kubric, image_set='validation')
