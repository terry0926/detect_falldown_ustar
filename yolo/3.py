from ultralytics import YOLO
import cv2
import math
from rpt import CoordinateTransformer
import json
from scipy.spatial.transform import Rotation as Rot
import numpy as np

with open(r'C:\Users\huang\Desktop\py\ustar\.venv\yolo\calib_general_webcam__1b3f_2247__640 (1).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

'''
旋轉向量 (rvec):
[[-1.18297896]
 [-0.02687296]
 [-0.02842615]]
位移向量 (tvec，單位同 obj_points):
[[-76.93519522]
 [ -2.51139401]
 [423.41285744]]
'''

r_obj = Rot.from_euler('xyz', [-1.18297896, -0.02687296, -0.02842615], degrees=True)
R = r_obj.as_matrix()
t_ext = np.array([-76.93519522, -2.51139401, 423.41285744], dtype=np.float32)
z_axis = [0, 0, 1]

worker = CoordinateTransformer(data, R, t_ext)
model = YOLO(r"C:\Users\huang\Desktop\py\ustar\.venv\yolo\yolo11n-pose_openvino_model")

results = model.predict(source="1", stream=True, show=False, device="cpu")

for r in results:
    frame = r.plot()

    if len(r.keypoints) > 0:
        pts = r.keypoints.xy[0].cpu().numpy()

        ls_x, ls_y = pts[5]
        rs_x, rs_y = pts[6]
        lh_x, lh_y = pts[15]
        rh_x, rh_y = pts[16]

        mid_shoulder_x = int((ls_x + rs_x) / 2)
        mid_shoulder_y = int((ls_y + rs_y) / 2)
        mid_hip_x = int((lh_x + rh_x) / 2)
        mid_hip_y = int((lh_y + rh_y) / 2)

        hip_dir = worker.worlde_direction(mid_hip_x, mid_hip_y)
        shoulder_dir = worker.worlde_direction(mid_shoulder_x, mid_shoulder_y)
        angle_deg = worker.get_vector_angle(hip_dir, shoulder_dir)

        # 淇�姝� 1锛歛b_len 鏀圭偤銆岀浉姗熶腑蹇冦€嶅埌銆岄珫閮ㄨ惤鍦伴粸銆嶇殑璺濋洟
        ab = worker.get_xyz(mid_hip_x, mid_hip_y, 0)
        cam = worker.cam_center.flatten()
        ab_len = worker.get_vector_length(*(ab - cam))

        # 淇�姝� 3锛歴olve_ac 鍙栧悎鐞嗘�ｅ€艰В
        candidates = worker.solve_ac(ab_len, 172, angle_deg)
        ac_len = max(float(c) for c in candidates if str(c).replace('.','',1).isdigit() and float(c) > 0)
        if isinstance(candidates, (list, tuple)):
            valid_candidates = [float(c) for c in candidates if isinstance(c, (int, float)) and c > 0]

        # 2. 安全取得 ac_len
        if valid_candidates:
            ac_len = max(valid_candidates)
        else:
            # 如果無解，設定一個合理的軀幹預設長度 (單位需與 1500 一致，假設是 mm)
            ac_len = 600.0 
            # print(f"無合理三角解: ab={ab_len:.1f}, angle={angle_deg:.1f}, 使用預設值")
        shoulder_xyz = worker.get_xyz_by_distance(mid_shoulder_x, mid_shoulder_y, ac_len)

        body_vec = shoulder_xyz - ab
        lean_angle = worker.get_vector_angle(body_vec, z_axis)

        cv2.line(frame, (mid_shoulder_x, mid_shoulder_y), (mid_hip_x, mid_hip_y), (0, 255, 0), 3)
        cv2.circle(frame, (mid_shoulder_x, mid_shoulder_y), 6, (0, 0, 255), -1)
        cv2.circle(frame, (mid_hip_x, mid_hip_y), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"lean angle: {np.round(lean_angle, 2)} deg", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("p", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()