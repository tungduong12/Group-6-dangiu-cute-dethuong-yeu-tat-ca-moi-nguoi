import cv2

# Khởi tạo bộ nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đọc ảnh hoặc video từ webcam
cap = cv2.VideoCapture(0)  # 0 là index của webcam, nếu bạn có nhiều webcam, có thể thử các giá trị khác nhau

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()

    # Chuyển ảnh sang đen trắng để tăng hiệu suất
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Vẽ hình chữ nhật xung quanh khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Hiển thị kết quả
    cv2.imshow('Face Detection', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ khi kết thúc
cap.release()
cv2.destroyAllWindows()
