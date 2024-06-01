Dưới đây là một phiên bản hoàn thiện cho mô tả về việc sử dụng CNN, lọc màu và YOLOv5 trong dự án nhận diện biển báo giao thông:

---

## Nhận diện Biển báo Giao thông với CNN, Lọc Màu và YOLOv5

![image](https://github.com/PVL-Linh/Traffic_Sign_Recognition_NhanDienBienBaoGiaoThong/assets/136146829/5d558b04-06ac-49f1-a098-90a0f855c83b)

### Mô tả Dự án

Trong dự án này, chúng tôi sử dụng mạng nơ-ron tích chập (CNN) để huấn luyện mô hình nhận diện biển báo giao thông từ ảnh. Sau đó, chúng tôi sử dụng phương pháp lọc màu để phát hiện biển báo từ các màu sắc phổ biến như đỏ, vàng và xanh. Cuối cùng, chúng tôi triển khai mô hình YOLOv5 để phát hiện biển báo trong video.

### Các Bước Thực Hiện

1. **Huấn luyện Mô hình CNN:**
   - Sử dụng mạng nơ-ron tích chập để huấn luyện mô hình nhận diện biển báo giao thông từ tập dữ liệu ảnh đã được gán nhãn.
   - Cải thiện hiệu suất của mô hình bằng cách điều chỉnh siêu tham số và kiến trúc mạng.

2. **Sử dụng Lọc Màu Đỏ, Vàng và Xanh:**
   - Áp dụng các bộ lọc màu đỏ, vàng và xanh để phát hiện các biển báo giao thông từ ảnh.
   - Sử dụng phương pháp phân đoạn hoặc kỹ thuật xử lý ảnh để tách biển báo từ nền.

3. **Triển Khai YOLOv5 cho Video:**
   - Sử dụng mô hình YOLOv5 để phát hiện biển báo giao thông trong video.
   - Cấu hình và huấn luyện mô hình YOLOv5 cho phù hợp với yêu cầu của dự án.

### Cách Triển Khai

1. **Triển Khai Mô hình CNN:**
   - Sử dụng một framework deep learning như TensorFlow hoặc PyTorch để huấn luyện và triển khai mô hình CNN.
   - Chạy mô hình trên dữ liệu thử nghiệm để đánh giá hiệu suất và chính xác.

2. **Lọc Màu:**
   - Sử dụng các thư viện xử lý ảnh như OpenCV để thực hiện lọc màu.
   - Điều chỉnh các tham số lọc để tối ưu hiệu suất phát hiện.

3. **Triển Khai YOLOv5:**
   - Sử dụng mã nguồn mở của YOLOv5 để triển khai mô hình trên video.
   - Đảm bảo cấu hình và cài đặt môi trường phù hợp cho YOLOv5.

### Kết Luận

Dự án này cung cấp một giải pháp toàn diện cho việc nhận diện biển báo giao thông từ ảnh và video. Sử dụng mạng nơ-ron tích chập, lọc màu và mô hình YOLOv5, chúng tôi có thể xây dựng một hệ thống nhận diện chính xác và hiệu quả cho các ứng dụng liên quan đến an toàn giao thông và tự động hóa lái xe.

--- 
