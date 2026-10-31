from ultralytics import YOLO
import cv2
import math
from rpt import CoordinateTransformer
import json
from scipy.spatial.transform import Rotation as Rot
import numpy as np

with open(r'C:\Users\huang\Desktop\py\ustar\.venv\yolo\calib_general_webcam__1b3f_2247__640 (1).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

r_obj = Rot.from_euler('xyz', [0, 0, 0], degrees=True)
R = r_obj.as_matrix()
t_ext = np.array([0, 0, 0])
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
        
        lh_x, lh_y = pts[11]
        rh_x,rh_y = pts[12]
        
        mid_shoulder_x = int((ls_x + rs_x) / 2)
        mid_shoulder_y = int((ls_y + rs_y) / 2)
        
        mid_hip_x = int((lh_x + rh_x) / 2)
        mid_hip_y = int((lh_y + rh_y) / 2)
        
        hip_dir=worker.worlde_direction(mid_hip_x, mid_hip_y)
        shoulder_dir=worker.worlde_direction(mid_shoulder_x, mid_shoulder_y)
        angle_deg=worker.get_vector_angle(hip_dir, shoulder_dir)
        ab=worker.get_xyz(mid_hip_x, mid_hip_y,0)
        ab_len=worker.get_vector_length(ab[0], ab[1], ab[2])
        ac_len=worker.solve_ac(ab_len,1.7,angle_deg)
        shoulder_z=worker.get_xyz_by_distance(mid_shoulder_x, mid_shoulder_y, ac_len[0])
        shoulder_xyz=worker.get_xyz(mid_shoulder_x, mid_shoulder_y,shoulder_z[2])
        body_vec=shoulder_xyz-ab
        lean_angle = worker.get_vector_angle(body_vec, z_axis)
        

        
        cv2.line(frame, (int(mid_shoulder_x), int(mid_shoulder_y)), (int(mid_hip_x), int(mid_hip_y)), (0, 255, 0), 3)
        cv2.circle(frame, (mid_shoulder_x, mid_shoulder_y), 6, (0, 0, 255), -1)
        cv2.circle(frame, (mid_hip_x, mid_hip_y), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"lean angle: {np.round(lean_angle, 2)} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("p", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
