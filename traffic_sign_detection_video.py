import cv2
import numpy as np

import cv2
import numpy as np
import torch
import torch.nn as nn

# Tạo lại kiến ​​trúc của mô hình bằng cách sử dụng Sequential
model = nn.Sequential(
    nn.Conv2d(3, 16, (2, 2), (1, 1), 'same'),
    nn.BatchNorm2d(16),
    nn.ReLU(True),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(16, 32, (2, 2), (1, 1), 'same'),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(32, 64, (2, 2), (1, 1), 'same'),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.Linear(1024, 256),
    nn.ReLU(True),
    nn.Linear(256, 43)
)

# Load trọng số của mô hình vào cấu trúc mô hình này
model_state_dict = torch.load("traffic_sign_model.pth", map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)
model.eval()

# Áp dụng phép tìm biển báo và tìm kiếm hộp bao biển báo
# Áp dụng phép tìm biển báo và tìm kiếm hộp bao biển báo
def detect_signs(image):
    # Chuyển đổi ảnh sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lower_threshold = int(np.mean(gray) - np.std(gray))
    upper_threshold = int(np.mean(gray) + np.std(gray))
    # Lọc ảnh để tìm biển báo giao thông
    mask = cv2.inRange(gray, lower_threshold, upper_threshold)

    # Tìm các đối tượng kết nối trong ảnh
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo hộp bao quanh các đối tượng
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    # Vẽ hộp bao quanh các đối tượng trên ảnh gốc và dự đoán biển báo
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Chuẩn bị dữ liệu đầu vào cho mô hình
        roi = gray[y:y + h, x:x + w]
        resized_roi = cv2.resize(roi, (32, 32))
        input_data = cv2.cvtColor(resized_roi, cv2.COLOR_GRAY2RGB)  # Chuyển đổi ảnh xám thành ảnh màu RGB
        input_data = np.expand_dims(input_data, axis=0).transpose(
            (0, 3, 1, 2)) / 255.0  # Thay đổi kích thước và thứ tự các chiều
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(-1)  # Thêm một chiều mới vào cuối cùng của tensor

        # Dự đoán biển báo
        with torch.no_grad():
            prediction = model(input_tensor.squeeze().unsqueeze(0))

        label = torch.argmax(prediction)
        confidence = torch.max(prediction)

        # Nếu độ chính xác dự đoán trên 90%, hiển thị nhãn dự đoán
        if confidence > 0.9:
            # Hiển thị nhãn dự đoán
            cv2.putText(image, f"Label: {label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image





# Đường dẫn đến video
video_path = "./img_test/traffic-sign-to-test.mp4"

# Mở video
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video đã mở thành công hay chưa
if not cap.isOpened():
    print("Không thể mở video.")

# Lấy thông tin về video (chiều rộng, chiều cao, số khung hình mỗi giây)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Tạo biến VideoWriter để ghi video kết quả
output_path = "./img_test/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Đọc từng khung hình trong video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Áp dụng phép tìm biển báo và tìm kiếm hộp bao biển báo
    sign_detection = detect_signs(frame)

    # Ghi khung hình kết quả vào video đích
    out.write(sign_detection)

    # Hiển thị khung hình gốc và kết quả phát hiện biển báo
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Sign Detection", sign_detection)

    # Phím 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(90) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()