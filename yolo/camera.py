
import cv2

cap = cv2.VideoCapture(0)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"解析度: {width} x {height}")
cap.release()