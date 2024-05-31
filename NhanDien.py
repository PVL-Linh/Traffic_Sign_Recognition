"""import cv2
import numpy as np
from keras.models import load_model
import os
import matplotlib.pyplot as plt
# Đường dẫn đến tệp hình ảnh
#image_path = "./Train-20240324T085203Z-001/Train/29_Maximum Speed 80/15_00056 - Copy (8).jpg"

image_path = './img_test/camDiBo_1.png'

# Đường dẫn đến tệp mô hình
model_path = "traffic_sign_model_2_64.h5"
file_path = './Train_copy'
def mapping_data(url = file_path):
    mapping = {}
    for file_img in os.listdir(file_path):
        number = 0
        chuoi = file_img.split("_")
        # Xử lý trường hợp số đứng đầu chuỗi
        if chuoi[0].isdigit():
            number = int(chuoi[0])
            mapping[number] = chuoi[1]
    return mapping

print (mapping_data(url=file_path))
# Kiểm tra xem tệp hình ảnh có tồn tại và có thể đọc được không
if cv2.haveImageReader(image_path):
    # Đọc hình ảnh từ đường dẫn
    image1 = cv2.imread(image_path)
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if image is not None:
        # Chuẩn bị hình ảnh cho mô hình (chú ý: bạn cần phải xử lý hình ảnh đầu vào để nó phù hợp với mô hình đã được huấn luyện trước đó)
        resized_image = cv2.resize(image, (32, 32 ))


        input_image = np.expand_dims(resized_image, axis=0)

        # Load mô hình từ tệp 'traffic_sign_model.h5'
        model = load_model(model_path)

        # Dự đoán từ hình ảnh
        predicted_number = model.predict(input_image)

        # In kết quả dự đoán
        print("Số dự đoán từ hình ảnh là:", predicted_number)
        # Chỉ mục của lớp có xác suất dự đoán cao nhất
        predicted_label_index = np.argmax(predicted_number)

        # In ra nhãn của lớp dự đoán
        print("Nhãn của lớp dự đoán:", predicted_label_index)


        # Chuyển đổi màu từ BGR sang RGB
        image_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        mapping = mapping_data()
        a = predicted_label_index
        label = mapping[a]


        # Hiển thị ảnh bằng Matplotlib
        plt.imshow(image_rgb)
        plt.title(label)
        plt.axis('off')  # Tắt trục
        plt.show()
    else:
        print("Không thể đọc hình ảnh từ đường dẫn đã cho.")
else:
    print("Không thể tìm thấy hoặc đọc được tệp hình ảnh. Vui lòng kiểm tra lại đường dẫn.")
"""


"""

import cv2
import numpy as np
from keras.models import load_model
import os
import matplotlib.pyplot as plt

def mapping_data(url):
    mapping = {}
    for file_img in os.listdir(url):
        number = 0
        chuoi = file_img.split("_")
        if chuoi[0].isdigit():
            number = int(chuoi[0])
            mapping[number] = chuoi[1]
    return mapping

def predict_traffic_sign(image_path, model, file_path):
    if cv2.haveImageReader(image_path):
        image1 = cv2.imread(image_path)
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if image is not None:
            resized_image = cv2.resize(image, (32, 32))
            input_image = np.expand_dims(resized_image, axis=0)
            predicted_number = model.predict(input_image)
            predicted_label_index = np.argmax(predicted_number)
            mapping = mapping_data(file_path)
            label = mapping[predicted_label_index]
            return image1, label
        else:
            print("Không thể đọc hình ảnh từ đường dẫn đã cho:", image_path)
            return None, None
    else:
        print("Không thể tìm thấy hoặc đọc được tệp hình ảnh:", image_path)
        return None, None

def main():
    # Đường dẫn đến thư mục chứa ảnh
    image_directory = './img_test/'
    # Đường dẫn đến tệp mô hình
    model_path = "traffic_sign_model_2.h5"
    # Đường dẫn đến thư mục chứa file ảnh và nhãn
    file_path = './Train_copy/'

    # Load mô hình từ tệp 'traffic_sign_model.h5'
    model = load_model(model_path)

    # Duyệt qua tất cả các tệp trong thư mục
    for filename in os.listdir(image_directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_directory, filename)
            image, label = predict_traffic_sign(image_path, model, file_path)
            if image is not None and label is not None:
                # Chuyển đổi màu từ BGR sang RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Hiển thị ảnh và nhãn bằng Matplotlib
                plt.imshow(image_rgb)
                plt.title(label)
                plt.axis('off')  # Tắt trục
                plt.show()

if __name__ == "__main__":
    main()
"""
import os
import cv2
import PIL
import torch
import torchvision as tv
import torch.nn as nn
import torchsummary as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


def predict_traffic_sign(image_path, model, label_names):
    if cv2.haveImageReader(image_path):
        image1 = cv2.imread(image_path)
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if image is not None:
            resized_image = cv2.resize(image, (32, 32))
            input_image = torch.tensor(np.expand_dims(resized_image, axis=0), dtype=torch.float32)
            predicted_number = model(input_image.unsqueeze(0)).detach().numpy()
            predicted_label_index = np.argmax(predicted_number)
            label = label_names[predicted_label_index]
            return image1, label
        else:
            print("Không thể đọc hình ảnh từ đường dẫn đã cho:", image_path)
            return None, None
    else:
        print("Không thể tìm thấy hoặc đọc được tệp hình ảnh:", image_path)
        return None, None

def main():
    # Đường dẫn đến thư mục chứa ảnh
    image_directory = './img_test/'
    # Đường dẫn đến tệp mô hình
    model_path = "traffic_sign_model.pth"
    # Đường dẫn đến thư mục chứa file ảnh và nhãn
    file_path = './Train_copy/'

    # Load mô hình từ tệp 'traffic_sign_model.pth'
    model = torch.load(model_path)

    # Đọc tệp CSV chứa thông tin về các nhãn
    labels_df = pd.read_csv('./Data/labels.csv')

    # Lấy tên nhãn bằng cách sử dụng index của label
    label_names = labels_df.iloc[:, 1].tolist()

    num_cols = 6  # Số cột trong lưới ảnh
    num_rows = 5  # Số hàng trong lưới ảnh
    num_images = num_cols * num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 15))

    image_index = 0

    for i in range(num_rows):
        for j in range(num_cols):
            filename = os.listdir(image_directory)[image_index]
            image_path = os.path.join(image_directory, filename)
            image, label = predict_traffic_sign(image_path, model, label_names)
            if image is not None and label is not None:
                # Thu nhỏ kích thước của ảnh
                resized_image = cv2.resize(image, (100, 100))
                # Hiển thị ảnh và nhãn trong lưới
                axes[i, j].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                axes[i, j].set_title(label)
                axes[i, j].axis('off')
                image_index += 1
                if image_index >= 30:
                    break
        if image_index >= 30:
            break

    # Ẩn các trục trống
    for ax in axes.flatten()[image_index:]:
        ax.axis('off')

    # Hiển thị lưới ảnh
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
