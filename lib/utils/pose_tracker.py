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

import os
import json
import shutil
import subprocess
import numpy as np
import os.path as osp
from scipy import signal


def run_openpose(
        video_file,
        output_folder,
        staf_folder,
        vis=False
):
    pwd = os.getcwd()

    os.chdir(staf_folder)

    render = 1 if vis else 0
    display = 2 if vis else 0
    cmd = [
        'build/examples/openpose/openpose.bin',
        '--model_pose', 'BODY_21A',
        '--tracking', '1',
        '--render_pose', str(render),
        '--video', video_file,
        '--write_json', output_folder,
        '--display', str(display)
    ]

    print('Executing', ' '.join(cmd))
    subprocess.call(cmd)
    os.chdir(pwd)


def read_posetrack_keypoints(output_folder, smoothen=False, smoothen_method='median'):

    people = dict()

    for idx, result_file in enumerate(sorted(os.listdir(output_folder))):
        json_file = osp.join(output_folder, result_file)
        data = json.load(open(json_file))
        # print(idx, data)
        for person in data['people']:
            person_id = person['person_id'][0]
            joints2d = person['pose_keypoints_2d']
            if person_id in people.keys():
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)
            else:
                people[person_id] = {
                    'joints2d': [],
                    'frames': [],
                }
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)

    for k in people.keys():
        # Convert to numpy array
        people[k]['joints2d'] = np.array(people[k]['joints2d'])
        if smoothen:
            people[k]['joints2d'] = smoothen_joints2d(people[k]['joints2d'], smoothen_method)
        else:
            people[k]['joints2d'] = people[k]['joints2d'].reshape((people[k]['joints2d'].shape[0], -1, 3))


        # people[k]['joints2d'] = people[k]['joints2d'].reshape(
        #     (len(people[k]['joints2d']), -1, 3))

        people[k]['frames'] = np.array(people[k]['frames'])

    return people


def smoothen_joints2d(joints, method):
    num_frames = joints.shape[0]

    # Reshape into 3D array with frames -> joints -> x, y, c
    joints = joints.reshape((num_frames, -1, 3))

    num_joints = joints.shape[1]

    # Split x and y coords from confidence value
    [coords, confs] = np.split(joints, [2], axis=2)

    # Test for the expected shapes
    assert(coords.shape == (num_frames, num_joints, 2))
    assert(confs.shape == (num_frames, num_joints, 1))

    if method is 'savgol':
        # Window size is capped by number of frames
        window_size = 7
        if window_size > num_frames:
            window_size = num_frames if num_frames % 2 is 1 else num_frames - 1

        # Polyorder must be less than window_size
        polyorder = min(window_size - 1, 5)

        # Smooth the coordinates
        coords_smooth = signal.savgol_filter(coords, window_size, polyorder, axis = 0)

        # Concat the coords with the confs
        return np.concatenate([coords_smooth, confs], axis = -1)
    else:
        # Split x and y
        [x, y] = list(map(lambda x: list(map(lambda xx: xx[0], x)), np.split(coords, [1], axis=1)))
        # Apply median filter
        x = signal.medfilt(x, [3])
        y = signal.medfilt(y, [3])

        coords_smooth = np.concatenate([np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))], axis = -1)

        assert(coords_smooth.shape == coords.shape)

        return np.concatenate([coords_smooth, confs], axis = -1)


def run_posetracker(video_file, staf_folder, posetrack_output_folder='/tmp', display=False):
    posetrack_output_folder = os.path.join(
        posetrack_output_folder,
        f'{os.path.basename(video_file)}_posetrack'
    )

    # run posetrack on video
    run_openpose(
        video_file,
        posetrack_output_folder,
        vis=display,
        staf_folder=staf_folder
    )

    people_dict = read_posetrack_keypoints(posetrack_output_folder)

    shutil.rmtree(posetrack_output_folder)

    return people_dict
