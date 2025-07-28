# 🚦 Nhận diện Biển báo Giao thông với CNN, Lọc Màu và YOLOv5

![image](https://github.com/PVL-Linh/Traffic_Sign_Recognition_NhanDienBienBaoGiaoThong/assets/136146829/5d558b04-06ac-49f1-a098-90a0f855c83b)

## 📌 Mô tả Dự án

Dự án này kết hợp giữa học sâu (deep learning) và xử lý ảnh truyền thống để nhận diện biển báo giao thông từ ảnh và video. Cụ thể:

* Huấn luyện mô hình **CNN** để phân loại biển báo.
* Sử dụng **lọc màu đỏ, vàng, xanh** nhằm phát hiện vùng nghi ngờ chứa biển báo.
* Áp dụng **YOLOv5** để phát hiện biển báo trong video theo thời gian thực.

---

## 🧠 Các Bước Thực Hiện

### 1️⃣ Huấn luyện Mô hình CNN

* Huấn luyện từ tập dữ liệu ảnh biển báo đã gán nhãn.
* Dùng PyTorch hoặc TensorFlow để xây dựng mô hình.
* Tối ưu hóa bằng điều chỉnh số lớp, hàm kích hoạt, learning rate...

### 2️⃣ Lọc Màu Đỏ, Vàng và Xanh

* Dùng OpenCV để lọc theo khoảng HSV tương ứng.
* Áp dụng phân đoạn ảnh để tách vùng có khả năng chứa biển báo.

### 3️⃣ Nhận diện bằng YOLOv5

* Sử dụng mô hình YOLOv5 đã tinh chỉnh để phát hiện biển báo trong video.
* Cấu hình lại anchor boxes, augment dữ liệu để phù hợp với đặc trưng biển báo.

---

## 🛠️ Cách Triển Khai

### ▶️ CNN:

```bash
# Cài đặt môi trường
pip install tensorflow keras

# Huấn luyện mô hình
python train_cnn.py
```

### ▶️ Lọc Màu:

```python
import cv2
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask_red = cv2.inRange(hsv, lower_red, upper_red)
```

### ▶️ YOLOv5:

```bash
# Clone YOLOv5
https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

# Huấn luyện mô hình
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt
```

---

## 🔍 Kết Luận

Dự án là một giải pháp toàn diện cho bài toán nhận diện biển báo giao thông với các ưu điểm:

✅ Chính xác cao nhờ học sâu (CNN, YOLOv5)
✅ Phát hiện nhanh theo thời gian thực
✅ Kết hợp truyền thống (lọc màu) và hiện đại (deep learning)\\

Phù hợp với các ứng dụng an toàn giao thông, hệ thống hỗ trợ lái xe thông minh, và nghiên cứu học thuật.

---

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp từ cộng đồng!
Bạn có thể:

* Mở **Issue** nếu gặp lỗi
* Tạo **Pull Request** để cải tiến mô hình hoặc giao diện

---

## 📫 Liên hệ

📧 Email: [phamvanlinh.sibinh2@gmail.com](phamvanlinh.sibinh2@gmail.com)
🌐 GitHub: [github.com/PVL-Linh](https://github.com/PVL-Linh)

---

⭐ Nếu bạn thấy dự án hữu ích, hãy **Star** và **Fork** để ủng hộ chúng tôi!
