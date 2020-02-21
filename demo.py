# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import argparse
# import colorsys
import os
import pickle
import shutil
import time

# import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.data_utils.kp_utils import convert_kps
from lib.dataset.inference import Inference
from lib.models.vibe import VIBE_Demo
from lib.utils.demo_utils import (convert_crop_cam_to_orig_img, download_ckpt,
                                  download_youtube_clip, images_to_video,
                                  prepare_rendering_results, smplify_runner,
                                  video_to_images)
from lib.utils.pose_tracker import run_posetracker
from lib.utils.renderer import Renderer
from multi_person_tracker import MPT

os.environ['PYOPENGL_PLATFORM'] = 'egl'


# Minimum number of frames that a person should be in picture before adding him
MIN_NUM_FRAMES = 15
# List of video files that should be processed
VIDEOS = ['videos/dance.mp4', 'videos/gta.mp4', 'videos/mma.mp4']


def main(args):

    for video_file in args.videos.split(','):
        if not os.path.isfile(video_file):
            print(f'Input video \"{video_file}\" does not exist!')
        else:
            run_vibe(video_file, args)


def run_vibe(video_file, args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Make output dirs
    output_path = os.path.join(
        args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)

    # Convert video to images
    image_folder, num_frames, img_shape = video_to_images(
        video_file, return_info=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    if not os.path.isabs(video_file):
        video_file = os.path.join(os.getcwd(), video_file)

    tracking_results = run_posetracker(
        video_file, staf_folder=args.staf_dir, display=args.display, smoothen=args.smoothen, smoothen_method=args.smoothen_method)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    print(f'Running VIBE on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}
    for person_id in tqdm(list(tracking_results.keys())):

        joints2d = tracking_results[person_id]['joints2d']
        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=None,
            joints2d=joints2d
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(
            dataset, batch_size=args.vibe_batch_size, num_workers=16)

        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [
            ], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(
                    batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(
                    batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:, :, 3:75].reshape(
                    batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :, 75:].reshape(
                    batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(
                    batch_size * seqlen, -1, 3))

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        norm_joints2d = np.concatenate(norm_joints2d, axis=0)
        norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
        norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

        # Run Temporal SMPLify
        update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
            new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                pred_rotmat=pred_pose,
                pred_betas=pred_betas,
                pred_cam=pred_cam,
                j2d=norm_joints2d,
                device=device,
                batch_size=norm_joints2d.shape[0],
                pose2aa=False,
            )

        # update the parameters after refinement
        print(
            f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
        pred_verts = pred_verts.cpu()
        pred_cam = pred_cam.cpu()
        pred_pose = pred_pose.cpu()
        pred_betas = pred_betas.cpu()
        pred_joints3d = pred_joints3d.cpu()
        pred_verts[update] = new_opt_vertices[update]
        pred_cam[update] = new_opt_cam[update]
        pred_pose[update] = new_opt_pose[update]
        pred_betas[update] = new_opt_betas[update]
        pred_joints3d[update] = new_opt_joints3d[update]


        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        vibe_results[person_id] = output_dict

    del model

    end = time.time()
    fps = num_frames / (end - vibe_time)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(
        f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(
        f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    print(
        f'Saving output results to \"{os.path.join(output_path, "vibe_output.pkl")}\".')

    # joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))
    for person in vibe_results.keys():
        dump_path = os.path.join(output_path, "vibe_output_%s.pkl" % person)
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        pickle.dump(vibe_results[person], open(dump_path, 'wb'))

    # if not args.no_render:
    #     # ========= Render results as a single video ========= #
    #     renderer = Renderer(resolution=(orig_width, orig_height),
    #                         orig_img=True, wireframe=args.wireframe)

    #     output_img_folder = f'{image_folder}_output'
    #     os.makedirs(output_img_folder, exist_ok=True)

    #     print(f'Rendering output video, writing frames to {output_img_folder}')

    #     # prepare results for rendering
    #     frame_results = prepare_rendering_results(vibe_results, num_frames)
    #     mesh_color = {k: colorsys.hsv_to_rgb(
    #         np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

    #     image_file_names = sorted([
    #         os.path.join(image_folder, x)
    #         for x in os.listdir(image_folder)
    #         if x.endswith('.png') or x.endswith('.jpg')
    #     ])

    #     for frame_idx in tqdm(range(len(image_file_names))):
    #         img_fname = image_file_names[frame_idx]
    #         img = cv2.imread(img_fname)

    #         for person_id, person_data in frame_results[frame_idx].items():
    #             frame_verts = person_data['verts']
    #             frame_cam = person_data['cam']

    #             mc = mesh_color[person_id]

    #             mesh_filename = None

    #             img = renderer.render(
    #                 img,
    #                 frame_verts,
    #                 cam=frame_cam,
    #                 color=mc,
    #                 mesh_filename=mesh_filename,
    #             )

    #         cv2.imwrite(os.path.join(output_img_folder,
    #                                  f'{frame_idx:06d}.png'), img)

    #         if args.display:
    #             cv2.imshow('Video', img)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break

    #     if args.display:
    #         cv2.destroyAllWindows()

    #     # ========= Save rendered video ========= #
    #     vid_name = os.path.basename(video_file)
    #     save_name = f'{vid_name.replace(".mp4", "")}_vibe_result.mp4'
    #     save_name = os.path.join(output_path, save_name)
    #     print(f'Saving result video to {save_name}')
    #     images_to_video(img_folder=output_img_folder,
    #                     output_vid_file=save_name)
    #     shutil.rmtree(output_img_folder)

    shutil.rmtree(image_folder)
    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--videos', type=str, default=','.join(VIDEOS),
                        help='input video paths')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--staf_dir', type=str, default='/content/VIBE/openpose',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--smoothen', type=bool, default=False,
                        help='Smoothen the 2D keypoints')

    parser.add_argument('--smoothen_method', type=str, default='median',
                        help='smoothen method to use. can be median or savgol')


    args = parser.parse_args()

    main(args)
