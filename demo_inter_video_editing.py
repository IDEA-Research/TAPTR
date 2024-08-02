import argparse
import gradio as gr
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from scipy.spatial.distance import cdist
import os
from torch.cuda.amp import autocast

from models.dino.taptr import build_taptr
from datasets.kubric import make_temporal_transforms, get_aux_target_hacks_list
from datasets.tapvid import resize_video


def downsample_points(points, min_d):
    # 计算每个点之间的距离
    distances = cdist(points, points)

    # 选择第一个点作为采样点
    sampled_points = [points[0]]
    sampled_indexes = [0]  # 第一个点在原序列中的索引为0
    last_point_index = 0

    # 选择下一个点，并确保与已选择的点之间的距离至少为1
    while last_point_index < len(points) - 1:
        min_distance = np.inf
        min_distance_index = -1

        # 找到距离上一个已选择点距离最近的且距离大于等于1的点
        for i in range(last_point_index + 1, len(points)):
            distance = np.min(distances[i, :last_point_index + 1])
            if distance >= min_d and distance < min_distance:
                min_distance = distance
                min_distance_index = i

        # 如果找到满足条件的点，则添加到结果列表中
        if min_distance_index != -1:
            sampled_points.append(points[min_distance_index])
            sampled_indexes.append(min_distance_index)
            last_point_index = min_distance_index
        else:
            # 如果没有找到满足条件的点，则跳出循环
            break

    return np.array(sampled_points), np.array(sampled_indexes)


def max_min_distance_sampling(points, M):
    selected_indexes = [np.random.choice(len(points))]  # 随机选择一个点的索引作为初始点
    selected_points = [points[selected_indexes[0]]]  # 将初始点添加到已选择点列表中

    while len(selected_points) < M:
        max_min_distance = -1
        new_index = None

        # 遍历所有点，找到与已选择点的最小距离最大的点
        for i, point in enumerate(points):
            if i in selected_indexes:
                continue  # 跳过已选择的点
            min_distance = min(np.linalg.norm(point - selected) for selected in selected_points)
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                new_index = i

        # 将找到的新点的索引和坐标添加到相应列表中
        selected_indexes.append(new_index)
        selected_points.append(points[new_index])

    return np.array(selected_points), np.array(selected_indexes)


def get_args():
    from main import get_args_parser
    from util.slconfig import DictAction, SLConfig
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("--path_ckpt", type=str, default="./checkpoints/taptr.pth")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    args = parser.parse_args()
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    return args


class Demo():
    def __init__(self, args):
        self.args = args
        self.model, _, _ = build_taptr(self.args)
        self.load_model(self.args.path_ckpt)
        self.model.eval().cuda()
        self.aux_target_hacks = get_aux_target_hacks_list("val", args)
        self.transforms            = make_temporal_transforms("val", fix_size=args.fix_size, strong_aug=False, args=args)
        self.num_query_per_video = args.num_queries_per_video_eval

    def load_model(self, path_ckpt):
        checkpoint = torch.load(path_ckpt, map_location="cpu")
        model_state_dict = {}
        for name, value in checkpoint["ema_model"].items():
            model_state_dict[name.replace("module.", "")] = value
        self.model.load_state_dict(model_state_dict)

    def process_one_video(self, video, fps, points, start_tracking_frames):
        """
        1. padding input points.
        2. sent to model to get results.
        3. plot the results in the video.
        return: the ploted video.
        Args:
            video (_type_): _description_
        """
        # TODO: process video using model.
        n_frames, height, width = video.shape[:3]  # video.shape[2], video.shape[1]
        if n_frames > 700:
            print("\nWarning, Too long video, clipping to 700 frames!"*10)
            video = video[:700] 
        video = torch.FloatTensor(video)
        input_video = torch.FloatTensor(resize_video(video.cpu().numpy(), (512, 512))).permute(0,3,1,2)
        points = np.array(points)[..., :2] / np.array([[width, height]]) * 512
        # points, p_idx = downsample_points(points, 1)
        if False:
            if len(points) > 256:
                # points, p_idx = max_min_distance_sampling(points, 256)
                p_idx = np.random.choice(len(points), 256, replace=False)
                points = points[p_idx]
            print("num points downsampled: ", len(points))
            start_tracking_frames = np.array(start_tracking_frames)[p_idx]
            queries = self.prepare_queries(512, 512, points.tolist(), n_frames, start_tracking_frames)
            input_video, queries_input, _ = self.align_format(input_video / 255.0, queries)
            input_video = input_video.cuda()
            queries_input = {k: v.cuda() for k, v in queries_input.items()}
            with torch.no_grad():
                outputs, queries_input_ = self.model([input_video], [queries_input])
            processed_video = self.plot_video(video, outputs, queries_input["num_real_pt"].cpu().item(), queries_input["query_frames"][:queries_input["num_real_pt"].cpu().item()].cpu())
        else:
            max_points = 256
            processed_video = video

            point_indexes = np.random.permutation(len(points))
            points = points[point_indexes]
            n_point_groups = len(points) // max_points // 8
            if n_point_groups * max_points * 8 < len(points):
                n_point_groups += 1
            for point_group_id in tqdm(range(n_point_groups)):
                print(f"processing {point_group_id} / {n_point_groups}")
                points_group = points[point_group_id*max_points:(point_group_id+1)*max_points]
                points_ids_group = point_indexes[point_group_id*max_points:(point_group_id+1)*max_points]
                start_tracking_frames_group = np.array(start_tracking_frames)[points_ids_group]
                queries_group = self.prepare_queries(512, 512, points_group.tolist(), n_frames, start_tracking_frames_group)
                input_video_group, queries_group, _ = self.align_format(input_video / 255.0, queries_group)
                input_video_group = input_video_group.cuda()
                queries_group = {k: v.cuda() for k, v in queries_group.items()}
                with torch.no_grad():
                    with autocast():
                        outputs, queries_input_ = self.model([input_video_group], [queries_group])
                processed_video = self.plot_video(processed_video, outputs, queries_group["num_real_pt"].cpu().item(), queries_group["query_frames"][:queries_group["num_real_pt"].cpu().item()].cpu())

        save_video_path = get_video_name()
        save_video(processed_video, save_video_path, fps)
        return save_video_path

    def align_format(self, images, targets):
        if self.transforms is not None:
            images, targets = self.transforms(images, targets)

        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                targets, images = hack_runner(targets, img=images)
        seq_name = targets.pop("seq_name")
        targets = {
            "pt_boxes": targets.pop("boxes"),
            "pt_labels": targets.pop("labels"),
            "pt_tracking_mask": targets.pop("tracking_mask"),
            "num_real_pt": targets.pop("num_real_pt"),
            "query_frames": targets.pop("query_frames"),
        }
        return images, targets, seq_name
        
    def prepare_queries(self, width, height, points, n_frames, start_tracking_frames, num_padding=-1):
        """prepare the points to be tracked.

        Args:
            width (int): width of the video
            height (int):  height of the video
            points (list of list): N, 6(xy, ...)

        Returns:
            _type_: _description_
        """
        query_xy = np.array(points)[:, :2]
        num_query_ = query_xy.shape[0]
        if (self.num_query_per_video > 0):
            np.random.seed(0)  # for reproducibility
            # xy
            margin_x = width // 16
            margin_y = height // 16
            min_distance = 1
            num_padding = self.num_query_per_video - query_xy.shape[0]
            # distant points
            if num_padding > 0:
                def generate_coordinates(original_coords, N, lower_bound=0, upper_bound=255, distance=3):
                    coordinates = original_coords.tolist()
                    while len(coordinates) < N:
                        new_coordinate = np.random.randint(lower_bound, upper_bound+1, size=2)
                        if all(np.linalg.norm(new_coordinate - c) >= distance for c in coordinates):
                            coordinates.append(new_coordinate)
                    return np.array(coordinates[-N:])
                padding_query_xy = generate_coordinates(query_xy, num_padding, lower_bound=margin_x, upper_bound=width-1-margin_x, distance=min_distance)
                padding_query_xy = padding_query_xy[:,None].repeat(n_frames, axis=1)
                query_xy = query_xy[:,None].repeat(n_frames, axis=1)
                query_xy = np.concatenate((query_xy, padding_query_xy), axis=0)
                # occ
                padding_query_occ = np.ones((num_padding, n_frames)) < 0
                # random restart all
                num_random_occ_frames = min(12, n_frames)
                padding_query_occ[:, :num_random_occ_frames] = \
                    padding_query_occ[:, :num_random_occ_frames] | \
                    np.arange(num_random_occ_frames)[None,:].repeat(padding_query_occ.shape[0], 0) < np.random.randint(0, num_random_occ_frames, padding_query_occ.shape[0])[:,None]
                query_occ = np.arange(n_frames)[None,:].repeat(num_query_, 0)
                query_occ = query_occ < np.array(start_tracking_frames)[:, None]
                query_occ = np.concatenate((query_occ, padding_query_occ), axis=0)
            else:
                query_occ = np.arange(n_frames)[None,:].repeat(num_query_, 0)
                query_occ = query_occ < start_tracking_frames[:, None]
                query_xy = query_xy[:,None].repeat(n_frames, axis=1)
        start_tracking_frames = torch.argmax(torch.tensor(1-query_occ), axis=1)
        tracking_mask = torch.arange(n_frames)[None, :].repeat(query_xy.shape[0], 1)
        tracking_mask = tracking_mask >= start_tracking_frames[:, None]
        queries = {
                "points": torch.from_numpy(query_xy).float(),
                'occluded': torch.from_numpy(query_occ), 
                'num_frames': n_frames, 
                'sampled_frame_ids': torch.arange(n_frames), 
                'tracking_mask': tracking_mask,
                'query_frames': start_tracking_frames,
                'sampled_point_ids': torch.arange(query_xy.shape[0]),
                "num_real_pt": torch.tensor([num_query_]),
                'seq_name': "demo_video",
            }
        return queries

    def plot_video(self, video, outputs, num_queries, start_tracking_frames):
        """plot the outputs on the video.

        Args:
            video (torch.tensor): len h w 3
            outputs (_type_): _description_
            num_queries (_type_): _description_
        """
        threshold_occ = 0.5
        len_temp, H, W, _ = video.shape
        pred_visibilities = outputs["full_seq_output"]["pred_logits"][0][:, :num_queries]  # len_temp, n_query, 3
        pred_occluded = pred_visibilities[..., 1].sigmoid() > threshold_occ  # n_query, len_temp
        pred_locations    = outputs["full_seq_output"]["pred_boxes"][0, :, :num_queries, :2]  # len_temp, n_query, 2
        pred_locations = pred_locations * torch.tensor([W, H]).float().cuda()[None, None, :]  # len_temp, n_query, 2
        video_plotted = draw_tracks_on_video(video, pred_locations, pred_occluded, start_tracking_frames, mode="cool", tracks_leave_trace=0)
        return video_plotted


def draw_tracks_on_video(
    video: torch.Tensor,
    tracks: torch.Tensor,
    visibility: torch.Tensor = None,
    query_frames: int = 0,
    compensate_for_camera_motion=False,
    mode = "rainbow",
    tracks_leave_trace = -1,
):
    """_summary_

    Args:
        video (torch.Tensor): len_temp, H, W, 3
        tracks (torch.Tensor): len_temp, n_query, 2
        visibility (torch.Tensor, optional): len_temp, n_query
        segm_mask (torch.Tensor, optional): _description_. Defaults to None.
        query_frame (int, optional): _description_. Defaults to 0.
        compensate_for_camera_motion (bool, optional): _description_. Defaults to False.
        mode (str, optional): _description_. Defaults to "rainbow".
        tracks_leave_trace (int, optional): _description_. Defaults to -1.
    """
    
    def _draw_pred_tracks(
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x N x 2
        draw_flag: np.ndarray,  # N
        vector_colors: np.ndarray,
        alpha: float = 0.5,
        tracks_leave_trace: int = 0,
        point_size = 2,
    ):
        T, N, _ = tracks.shape

        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                if not draw_flag[i]:
                    continue
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].tolist(),
                        point_size,
                        cv2.LINE_AA,
                    )
            if tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb
    
    if mode == "rainbow":
        color_map = cm.get_cmap("gist_rainbow")
    elif mode == "cool":
        color_map = cm.get_cmap(mode)
        
    T, H, W, C = video.shape
    _, N, D = tracks.shape

    assert D == 2
    assert C == 3
    video = video.byte().detach().cpu().numpy()  # S, H, W, C
    tracks = tracks.long().detach().cpu().numpy()  # S, N, 2

    res_video = []

    # process input video
    for rgb in video:
        res_video.append(rgb.copy())

    vector_colors = np.zeros((T, N, 3))
    if mode == "rainbow":
        start_locations = torch.gather(torch.tensor(tracks), 0, query_frames[None, :, None].repeat(1,1,2))[0]
        y_min, y_max = (
            start_locations[..., 1].min(),
            start_locations[..., 1].max(),
        )
        norm = plt.Normalize(y_min, y_max)
        for n in range(N):
            color = color_map(norm(start_locations[n, 1]))
            color = np.array(color[:3])[None] * 255
            vector_colors[:, n] = np.repeat(color, T, axis=0)
    else:
        # color changes with time
        for t in range(T):
            # color = np.array(color_map(t / T)[:3])[None] * 255
            color = np.array(color_map(t)[:3])[None] * 255
            vector_colors[t] = np.repeat(color, N, axis=0)
    
    point_size = 1
    #  draw tracks
    if tracks_leave_trace != 0:
        for t in range(1, T):
            first_ind = (
                max(0, t - tracks_leave_trace)
                if tracks_leave_trace >= 0
                else 0
            )
            curr_tracks = tracks[first_ind : t + 1]
            curr_colors = vector_colors[first_ind : t + 1]
            if compensate_for_camera_motion:
                raise NotImplementedError("compensate for camera motion is not implemented.")
                # diff = (
                #     tracks[first_ind : t + 1, segm_mask <= 0]
                #     - tracks[t : t + 1, segm_mask <= 0]
                # ).mean(1)[:, None]

                # curr_tracks = curr_tracks - diff
                # curr_tracks = curr_tracks[:, segm_mask > 0]
                # curr_colors = curr_colors[:, segm_mask > 0]
            
            draw_flag = t >= query_frames
            res_video[t] = _draw_pred_tracks(
                res_video[t],
                curr_tracks,
                draw_flag,
                curr_colors,
                tracks_leave_trace=tracks_leave_trace,
                point_size=point_size,
            )

    #  draw points
    for t in range(T):
        for i in range(N):
            if t < query_frames[i]:
                continue
            coord = (tracks[t, i, 0], tracks[t, i, 1])
            visibile = True
            if visibility is not None:
                visibile = visibility[t, i]
            if coord[0] != 0 and coord[1] != 0:
                if not compensate_for_camera_motion:
                # or (
                #     compensate_for_camera_motion and segm_mask[i] > 0
                # )
                    if visibile:
                        cv2.circle(
                            res_video[t],
                            coord,
                            int(point_size * 2),
                            vector_colors[t, i].tolist(),
                            thickness=-1 if visibile else point_size,
                        )
                else:
                    raise NotImplementedError("compensate for camera motion is not implemented.")

    #  construct the final rgb sequence
    # if self.show_first_frame > 0:
    #     res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
    return torch.from_numpy(np.stack(res_video)).byte()


def get_video_name(dir_save="./saved_videos", prefix="processed_video_"):
    import os
    files = os.listdir(dir_save)
    file_orders = [int(file.split(".")[0][len(prefix):]) for file in files if file.endswith(".mp4") and file.startswith(prefix)]
    if len(file_orders) == 0:
        return os.path.join(dir_save, prefix + "0.mp4")
    else:
        return os.path.join(dir_save, prefix + str(max(file_orders) + 1) + ".mp4")


def save_video(video, save_path, fps):
    """save a video into a specified file.

    Args:
        video (torch.tensor): n h w c
        filename (string): _description_
    """
    wide_list = list(video.unbind(0))
    wide_list = [wide.cpu().numpy() for wide in wide_list]
    clip = ImageSequenceClip(wide_list, fps=fps)
    clip.write_videofile(save_path, codec="libx264", fps=fps, logger=None)
    print(f"Video saved to {save_path}")
    return save_path


def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames), fps


def video_to_interaction(input_video, frame_selector):
    """
    1. get the frame from input_video.
    2. get the interaction from the frame.
    3. return the interaction.
    Args:
        input_video (_type_): _description_
        frame_selector (_type_): _description_
        interaction (_type_): _description_
    """
    # frame_selector = int(700 * frame_selector)
    frame_selector = int(frame_selector)
    frames = cv2.VideoCapture(input_video)
    interaction = None
    # fps = int(frames.get(cv2.CAP_PROP_FPS))
    frame_id = 0
    if not frames.isOpened():
        print("Error opening video file")
    else:
        while frames.isOpened():
            ret, frame = frames.read()
            print("Getting the interaction frame: ", frame_id, frame_selector)
            if frame_id > 700:
                print(f"Too long video ({input_video}) for the demo! - video_to_interaction")
                return None
            if frame_id == frame_selector:
                interaction = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                break
            frame_id += 1
        frames.release()
    if interaction is None:
        raise ValueError("Frame not found")
    # return {"image": interaction, "points": []}
    mask_ori = np.zeros((*interaction.shape[:2], 4)).astype(np.uint8)
    interaction = np.concatenate([interaction, np.ones((*interaction.shape[:2], 1))*255], axis=-1).astype(np.uint8)
    interaction = {
        "layers": [mask_ori],
        "background": interaction,
        "composite": interaction,
    }
    return interaction


def process_one_video(input_video, frame, interaction):
    """tracking the points from interaction along the video.

    Args:
        input_video (_type_): _description_
        frame (_type_): _description_
        interaction (_type_): _description_

    Returns:
        _type_: _description_
    """
    global DemoCore
    # TODO: should we downsample the points?
    mask = interaction["layers"][0][..., -1] > 0
    points_y, points_x = np.where(mask)
    points = np.stack([points_x, points_y], axis=1)
    points = points.tolist()
    start_tracking_frames = [frame] * len(points)
    print("num points: ", len(points))
    # points = interaction["points"]
    if len(points) == 0:
        raise ValueError("No points found to be tracked.")
    video, fps = read_video_from_path(input_video)
    output_video = DemoCore.process_one_video(video, fps, points, start_tracking_frames)
    return output_video, interaction


with gr.Blocks(title="TAPTR") as demo:
    title_markdown = ("""
        # TAPTR: **T**racking **A**ny **P**oint with **TR**ansformer as Detection
        ### Area Tracking Demo [[PointTrajectory Demo]](https://taptr-trajectory.deepdataspace.com)
        [[Project Page]](https://taptr.github.io) [[Paper-TAPTR]](https://arxiv.org/abs/2403.13042)  [[Paper-TAPTRv2]](https://arxiv.org/abs/2407.16291) [[Code]](https://github.com/IDEA-Research/TAPTR)
    """)
    tips = ("""
        **Usage** \\
            1. Upload a video in Input Video module. \\
            2. Select a frame in Frame Selector module, the corresponding frame will be shown in Interaction module. (The first frame is selected by default.) \\
            3. Click the draw button in the Interaction module and draw the area where you want to edit. (Please adjust the brush thickness to be as fine as possible to reduce computational burden. By default, we will downsample the points within the editing area by about 8 times).\\
            4. Click on Submit button to start tracking the editing area. \\
            5. The output video will be shown in Output Video module. \\
        More details, please refer to the example video. \\
        **Note** \\
            1. TAPTR has broad application scenarios, such as slam, AR, motion capture, and video editing. If you have any collaboration intentions, please contact us. \\
            2. Limited by the hardware our demo running on, a video that is too long may result in the long machine occupation, so we reject the video longer than 700 frames. \\
            3. If the height of the video is too large, the brush option in Interaction module may be blocked, please resize your video and retry. \\
            4. If you have any questions feel free to contact us or open an issue in our [repo](https://github.com/IDEA-Research/TAPTR).
    """)
    notation = ("""
        💡Since TAPTR is a general point tracking method, feel free to upload and evaluate your own video. 
        
        💡To alleviate our computational load, if the length of your input video exceeds 700 frames, we will clamp the video to 700 frames.
                
        💡The frame selector can not adjust the range automatically, if the selected frame exceed the video length, the demo will fail. 
    """)
    gr.Markdown(title_markdown)
    with gr.Row():
        with gr.Column(scale=0.5):
            input_video = gr.Video(label="Input Video", height=400)
            frame_selector = gr.Slider(minimum=0, maximum=700, value=0, label="Frame Selector")
            submit_btn = gr.Button("Submit")
            if os.path.exists("./assets/example_videos/Box.mp4"):
                gr.Markdown(notation)
                with gr.Row():
                    gr.Examples(examples=[
                        [f"./assets/example_videos/Box.mp4"],
                    ], inputs=[input_video], label="Example-Box")
                    gr.Examples(examples=[
                        [f"./assets/example_videos/Sofa.mp4"],
                    ], inputs=[input_video], label="Example-Sofa")
                    gr.Examples(examples=[
                        [f"./assets/example_videos/RabbitAndYogurt.mp4"],
                    ], inputs=[input_video], label="Example-RabbitAndYogurt")
                    gr.Examples(examples=[
                        [f"./assets/example_videos/RollingBasketball.mp4"],
                    ], inputs=[input_video], label="Example-RollingBasketball")
            if os.path.exists("./assets/VideoEditing.mp4"):
                example_video = gr.Video(label="Usage", height=250, value="./assets/VideoEditing.mp4")
                gr.Markdown(tips)
        with gr.Column():
            interaction = gr.ImageMask(label="Interaction", type='numpy', interactive=True, height=700)
            output_video = gr.Video(label="Out Video", height=650)
    input_video.change(fn=video_to_interaction, inputs=[input_video, frame_selector], outputs=[interaction])
    frame_selector.change(fn=video_to_interaction, inputs=[input_video, frame_selector], outputs=[interaction])
    submit_btn.click(fn=process_one_video, inputs=[input_video, frame_selector, interaction], outputs=[output_video, interaction])
demo.queue()


if __name__ == "__main__":
    global DemoCore
    args = get_args()
    DemoCore = Demo(args)
    demo.launch(server_name="0.0.0.0", server_port=10003)

# CUDA_VISIBLE_DEVICES=0 python demo_inter_video_editing.py -c config/TAPTR.py --path_ckpt logs/TAPTR/taptr.pth