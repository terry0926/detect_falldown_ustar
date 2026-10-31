import cv2

# 初始化鏡頭與讀取迴圈
cap = cv2.VideoCapture(1)  # 開啟預設攝影機

while True:
    ret, frame = cap.read()  # 讀取當前影格
    cv2.imshow('Camera', frame)  # 顯示即時畫面
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 按下 s 鍵存檔
        cv2.imwrite('image.jpg', frame)  # 儲存影像
        print("影像已儲存")
    elif key == ord('q'):  # 按下 q 鍵退出
        break

cap.release()  # 釋放資源
cv2.destroyAllWindows()
