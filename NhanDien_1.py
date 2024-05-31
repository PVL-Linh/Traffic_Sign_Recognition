import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms

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


# Đọc tệp CSV chứa thông tin về các nhãn
labels_df = pd.read_csv('./Data/labels.csv')


# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    image = cv2.resize(image, (32, 32))  # Resize về (32, 32)
    # Chuyển đổi thành tensor và chuẩn hóa
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = preprocess(image)
    return image


# Dự đoán nhãn
def predict_label(image):
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Thêm chiều batch và dự đoán
        predicted_label_index = torch.argmax(output).item()
    return predicted_label_index


# Hàm chính
def main():
    # Đường dẫn đến ảnh cần dự đoán
    image_path = './img_test/stop_1.png'

    # Tiền xử lý ảnh
    image = preprocess_image(image_path)

    # Dự đoán nhãn
    label_index = predict_label(image)

    # Lấy tên nhãn từ dataframe
    label_name = labels_df.loc[labels_df['ClassId'] == label_index, 'Name'].values[0]

    # Hiển thị ảnh và nhãn dự đoán
    plt.imshow(image.permute(1, 2, 0))  # Chuyển tensor thành numpy array
    plt.title(f"Predicted Label: {label_name}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
