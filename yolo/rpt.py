import math

import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation as Rot

with open(r'C:\\Users\\huang\\Desktop\\py\\ustar\\.venv\\yolo\\calib_general_webcam__1b3f_2247__1920 (1).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

class CoordinateTransformer:
    def __init__(self, json_data, R, t):
        self.K = np.array(json_data['camera_matrix']['data']).reshape(3, 3)
        self.D = np.array(json_data['distortion_coefficients']['data'])
        self.K_inv = np.linalg.inv(self.K)
        # P_world = R_inv @ (P_cam - t)
        self.R_inv = np.linalg.inv(R)
        self.cam_center = -self.R_inv @ t.reshape(3, 1)
        
    def get_xyz(self, u, v, z):
        p_distorted = np.array([[[u, v]]], dtype=np.float32)
        p_undistorted = cv2.undistortPoints(p_distorted, self.K, self.D, P=self.K)
        u_c, v_c = p_undistorted[0][0]

        pixel_vec = self.K_inv @ np.array([u_c, v_c, 1.0]).reshape(3, 1)
        
        world_dir = self.R_inv @ pixel_vec

        #  cam_center_z + s * world_dir_z = target_z
        # s = (target_z - cam_center_z) / world_dir_z
        s = (z - self.cam_center[2, 0]) / world_dir[2, 0]
        
        world_xyz = self.cam_center + s * world_dir
        
        return world_xyz.flatten()
    
    def worlde_direction(self, u, v):
        p_distorted = np.array([[[u, v]]], dtype=np.float32)
        p_undistorted = cv2.undistortPoints(p_distorted, self.K, self.D, P=self.K)
        u_c, v_c = p_undistorted[0][0]
        pixel_vec = self.K_inv @ np.array([u_c, v_c, 1.0]).reshape(3, 1)
        
        world_dir = self.R_inv @ pixel_vec
        
        return world_dir.flatten()
    
    def get_vector_length(self, x, y, z):
        vec = np.array([x, y, z])
        return np.linalg.norm(vec)
    
    def get_vector_angle(self, v1, v2):

        v1 = np.array(v1)
        v2 = np.array(v2)
        
        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)
        
        if mag_v1 == 0 or mag_v2 == 0:
            return 0.0
        
        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def solve_ac(self, ab_len, bc_len, angle_a_deg):
        rad_a = math.radians(angle_a_deg)
        try:
            part1 = ab_len * math.cos(rad_a)
            part2_sq = bc_len**2 - (ab_len * math.sin(rad_a))**2
            
            if part2_sq < 0:
                return []  # 回傳空清單代表無解
                
            part2 = math.sqrt(part2_sq)
            res1 = part1 + part2
            res2 = part1 - part2
            return [res1, res2] # 統一回傳清單
        except:
            return [] # 出錯也回傳空清單

        
    def get_xyz_by_distance(self, u, v, L):
        world= self.worlde_direction(u, v)
        unit_dir = world / np.linalg.norm(world)
        world_xyz = self.cam_center.flatten() + L * unit_dir
        
        return world_xyz


