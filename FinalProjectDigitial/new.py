import cv2
import os
import cv2.data
import numpy as np
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


# Lấy id và tên người dùng từ người dùng và kiểm tra sự tồn tại của thư mục data/{user_id}
while True:
    user_id = str(input("Nhập ID: "))
    user_name = input("Nhập tên người dùng: ")
    user_data_dir = f'data/{user_id}/'
    if not os.path.exists(user_data_dir):
        break
    else:
        print("ID đã tồn tại, vui lòng nhập ID khác!")

# Tạo thư mục data/{user_id}/
if not os.path.exists(user_data_dir):
    os.makedirs(user_data_dir)

sampleNum = 0
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fa = detector.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in fa:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
        if not os.path.exists('data'):
            os.makedirs('data')
        sampleNum+=1
        cv2.imwrite(f'{user_data_dir}anh{sampleNum}.jpg', gray[y:y+h, x:x+w])
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
    if sampleNum > 200:
        break

# Lưu tên người dùng vào tệp labels.txt
with open('labels.txt', 'a') as f:
    f.write(f'{user_id}: {user_name}\n')
cap.release()
cv2.destroyAllWindows()

## Người dùng nhập id -> kiểm tra id có tồn tại hay chưa
## Đọc tên người dùng từ file labels.txt