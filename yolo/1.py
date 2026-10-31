from ultralytics import YOLO
import time

model = YOLO(r"C:\Users\huang\Desktop\py\ustar\.venv\yolo\yolo11n-pose_openvino_model")
results = model.predict(source="0", stream=True, show=True, device="cpu")

for r in results:
    print(r.keypoints[0].xy.cpu().numpy()[0][0][0])  
    print(r.keypoints[0].xy.cpu().numpy()[0][0][1])  
    
    pass
