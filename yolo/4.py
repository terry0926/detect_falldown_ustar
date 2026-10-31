from ultralytics import YOLO
import cv2
import math
from rpt import CoordinateTransformer
import json
from scipy.spatial.transform import Rotation as Rot
import numpy as np

# 載入內參
with open(r'C:\Users\huang\Desktop\py\ustar\.venv\yolo\calib_general_webcam__1b3f_2247__640 (1).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# --- 修正 1：rvec 轉換 (solvePnP 的 rvec 是旋轉向量，非歐拉角) ---
rvec = np.array([-1.18297896, -0.02687296, -0.02842615], dtype=np.float32)
r_obj = Rot.from_rotvec(rvec)  # 必須用 from_rotvec
R = np.array([
    [ 0.99932015, -0.0363624,   0.0060832 ],
    [-0.00811463, -0.37789097, -0.92581454],
    [ 0.03596363,  0.92513577, -0.37792913]
], dtype=np.float32)

t_ext = np.array([-76.93519522, -2.51139401, 423.41285744], dtype=np.float32)

z_axis = [0, 0, 1]
worker = CoordinateTransformer(data, R, t_ext)

model = YOLO(r"C:\Users\huang\Desktop\py\ustar\.venv\yolo\yolo11n-pose_openvino_model")
results = model.predict(source="1", stream=True, show=False, device="cpu")

for r in results:
    frame = r.plot()
    if len(r.keypoints) > 0:
        # 取得關鍵點
        pts = r.keypoints.xy[0].cpu().numpy()
        if len(pts) < 17: continue # 確保點數足夠
        
        ls_x, ls_y = pts[5] 
        rs_x, rs_y = pts[6]  
        lh_x, lh_y = pts[15] 
        rh_x, rh_y = pts[16] 

        mid_shoulder_x = int((ls_x + rs_x) / 2)
        mid_shoulder_y = int((ls_y + rs_y) / 2)
        mid_hip_x = int((lh_x + rh_x) / 2)
        mid_hip_y = int((lh_y + rh_y) / 2)

        # 向量與角度計算
        hip_dir = worker.worlde_direction(mid_hip_x, mid_hip_y)
        shoulder_dir = worker.worlde_direction(mid_shoulder_x, mid_shoulder_y)
        angle_deg = worker.get_vector_angle(hip_dir, shoulder_dir)

        # 計算 ab_len
        ab = worker.get_xyz(mid_hip_x, mid_hip_y, 0)
        cam = worker.cam_center.flatten()
        ab_len = worker.get_vector_length(*(ab - cam))

        # --- 修正 3：ac_len 安全判斷 (解決 ValueError) ---
        # bc_len 設為 172 (若單位是 cm) 或 1720 (若單位是 mm)，請與 t_ext 統一
        candidates = worker.solve_ac(ab_len, 140, angle_deg) 
        
        valid_candidates = []
        if isinstance(candidates, (list, tuple)):
            valid_candidates = [float(c) for c in candidates if isinstance(c, (int, float)) and c > 0]

        if valid_candidates:
            ac_len = min(valid_candidates)
        else:
            ac_len = ab_len # 預設軀幹長度 (cm)
            
        shoulder_xyz = worker.get_xyz_by_distance(mid_shoulder_x, mid_shoulder_y, ac_len)
        body_vec = shoulder_xyz - ab
        lean_angle = worker.get_vector_angle(body_vec, z_axis)

        cv2.line(frame, (mid_shoulder_x, mid_shoulder_y), (mid_hip_x, mid_hip_y), (0, 255, 0), 3)
        cv2.putText(frame, f"lean angle: {np.round(lean_angle, 2)} deg", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("p", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
