import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

camera_matrix = np.array([
    [695.222515, 0, 316.318780],
    [0, 691.118267, 272.275376],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([
    -0.416786, 0.226063, -0.006281, -0.008625, -0.061545
], dtype=np.float32)

# 單位 cm，地板上量的真實座標
obj_points = np.array([
    [0,9,108],
    [127,9,108],
    [0,   0,   0],
    [127, 0,   0],
    [0,   -88,  0],
    [127, -88,  0]
], dtype=np.float32)

# 對應的像素座標，順序要跟上面一致
img_points = np.array([
    [181,84],
    [404,84],
    [187,267],
    [400,265],
    [169,332],
    [418,329]
], dtype=np.float32)

success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

if success:
    R, _ = cv2.Rodrigues(rvec)
    cam_center = -np.linalg.inv(R) @ tvec
    
    print(f"相機高度 Z: {cam_center[2][0]:.2f} cm")
    
    print(f"\nR =\n{R}")
    print(f"\nt = {tvec.flatten()}")