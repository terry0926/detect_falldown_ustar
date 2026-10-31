import cv2
import csv

count = 0

def get_location(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: # 點擊左鍵
        print(f"點擊座標: x={x}, y={y}")
        # 在點擊處畫個紅點（選用）
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Image", img)
        global count
        count+=1
        
        with open('coordinates.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([count, x, y])

with open('coordinates.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['count','x','y'])
    
    
img = cv2.imread('image.jpg')
cv2.imshow("Image", img)

# 設定滑鼠回呼函數
cv2.setMouseCallback("Image", get_location)

cv2.waitKey(0)
cv2.destroyAllWindows()
