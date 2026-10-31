import numpy as np

def get_3d_point_on_floor(u, v, K, R, t, height):
    """
    將像素座標 (u, v) 轉換為世界座標 (X, Y, Z)
    假設該點位於 Z=0 的地板上
    """
    # 1. 轉為相機歸一化座標 (K 的逆矩陣)
    K_inv = np.linalg.inv(K)
    p_pixel = np.array([u, v, 1])
    p_cam_dir = K_inv @ p_pixel # 得到射線方向
    
    # 2. 轉到世界座標的方向 (R 的轉置)
    # 注意：世界座標方向 V = R^T * p_cam_dir
    v_world = R.T @ p_cam_dir
    
    # 3. 計算射線與 Z=0 平面的交點
    # 相機在世界座標的位置 C = -R^T @ t
    camera_pos = -R.T @ t
    cam_z = camera_pos[2]
    ray_z = v_world[2]
    
    # 比例因子 s：使得 camera_pos + s * v_world 的 Z 分量為 0
    s = -cam_z / ray_z
    
    p_world = camera_pos + s * v_world
    return p_world
