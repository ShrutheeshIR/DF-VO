import cv2
import copy
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn import linear_model
from time import time
from tqdm import tqdm

from libs.deep_depth.monodepth2 import Monodepth2DepthNet
from libs.geometry.ops_3d import *
from libs.general.frame_drawer import FrameDrawer
from libs.general.timer import Timers
from libs.matching.deep_flow import LiteFlow
from libs.camera_modules import SE3, Intrinsics
from libs.utils import *
from tool.evaluation.tum_tool.associate import associate, read_file_list
from tool.evaluation.tum_tool.pose_evaluation_utils import rot2quat

class VisualOdometry():
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration reading from yaml file
        """
        self.cam_intrinsics = Intrinsics()
        # predicted global poses
        self.global_poses = {0: SE3()}
        # tracking stage
        self.tracking_stage = 0
        # configuration
        self.cfg = cfg

        # visualization interface
        self.initialize_visualization_drawer()

        self.ref_data = {
                        'id': [],
                        'timestamp': {},
                        'img': {},
                        'depth': {},
                        'raw_depth': {},
                        'pose': {},
                        'kp': {},
                        'kp_best': {},
                        'kp_list': {},
                        'pose_back': {},
                        'kp_back': {},
                        'flow': {},  # from ref->cur
                        'flow_diff': {},  # flow-consistency-error of ref->cur
                        }
        self.cur_data = {
                        'id': 0,
                        'timestamp': 0,
                        'img': np.zeros(1),
                        'depth': np.zeros(1),
                        'pose': np.eye(4),
                        'kp': np.zeros(1),
                        'kp_best': np.zeros(1),
                        'kp_list': np.zeros(1),
                        'pose_back': np.eye(4),
                        'kp_back': np.zeros(1),
                        'flow': {},  # from cur->ref
                        }

    def get_intrinsics_param(self, dataset):
        """Read intrinsics parameters for each dataset
        Args:
            dataset (str): dataset
                - kitti
                - tum-1/2/3
        Returns:
            intrinsics_param (float list): [cx, cy, fx, fy]
        """
        # Kitti
        img_seq_dir = os.path.join(
                        self.cfg.directory.img_seq_dir,
                        self.cfg.seq
                        )
        intrinsics_param = load_kitti_odom_intrinsics(
                        os.path.join(img_seq_dir, "calib.txt"),
                        self.cfg.image.height, self.cfg.image.width
                        )[2]

        return intrinsics_param

    def get_img_depth_dir(self):
        """Get image data directory and (optional) depth data directory

        Returns:
            img_data_dir (str): image data directory
            depth_data_dir (str): depth data directory / None
            depth_src (str): depth data type
                - gt
                - None
        """
        # get image data directory
        img_seq_dir = os.path.join(self.cfg.directory.img_seq_dir, self.cfg.seq)
        img_data_dir = os.path.join(img_seq_dir, "image_2")