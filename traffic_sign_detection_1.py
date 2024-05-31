from tkinter.filedialog import askopenfilename

import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
import math
import numpy as np
from sympy import re



def show_images(images, cmap="viridis"):
    column = 3
    row = int(math.ceil(len(images)/column))
    plt.figure(figsize=(20, 10))
    for i, img in enumerate(images):
        plt.subplot(row,column,i+1)
        if cmap != "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img, cmap=cmap)
        plt.axis('off')

# Đọc hình ảnh

img111111 = '1.jpg'
"""bgr_images = [cv2.imread(str(name)) for name in image_names]
hsvs = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in bgr_images]"""
#bgr_images = cv2.imread(img111111)
#hsvs = cv2.cvtColor(bgr_images, cv2.COLOR_BGR2HSV)



"""a = askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
bgr_images = cv2.imread(a)
bgr_images = load_img(bgr_images)"""

def filter_signs_by_color(image):
    """Lọc các đối tượng màu đỏ và màu xanh dương - Có thể là biển báo.
        Ảnh đầu vào là ảnh màu BGR
    """
    # Chuyển ảnh sang hệ màu HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Lọc màu đỏ cho stop và biển báo cấm
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask_1 = cv2.inRange(image, lower1, upper1) # dải màu đỏ thứ nhất
    mask_2 = cv2.inRange(image, lower2, upper2) # dải màu đỏ thứ hai
    mask_r = cv2.bitwise_or(mask_1, mask_2) # kết hợp 2 kết quả từ 2 dải màu khác nhau

    # Lọc màu xanh cho biển báo điều hướng
    lower3, upper3 = np.array([85, 50, 200]), np.array([135, 250, 250])
    mask_b = cv2.inRange(image, lower3,upper3)

    # Kết hợp các kết quả
    mask_final  = cv2.bitwise_or(mask_r,mask_b)
    return mask_final

#masks = [filter_signs_by_color(img) for img in bgr_images]
#masks = filter_signs_by_color(bgr_images)

def get_boxes_from_mask(mask):
    """Tìm kiếm hộp bao biển báo
    """
    bboxes = []

    nccomps = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_height, im_width = mask.shape[:2]
    for i in range(numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # Lọc các vật quá nhỏ, có thể là nhiễu
        if w < 20 or h < 20:
            continue
        # Lọc các vật quá lớn
        if w > 0.8 * im_width or h > 0.8 * im_height:
            continue
        # Loại bỏ các vật có tỷ lệ dài / rộng quá khác biệt
        if w / h > 2.0 or h / w > 2.0:
            continue
        bboxes.append([x, y, w, h])
    return bboxes

#results = []
"""for i, img in enumerate(bgr_images):
    mask = filter_signs_by_color(img) # lọc theo màu sắc
    bboxes = get_boxes_from_mask(mask) # tìm kiếm khung bao của các vật từ mặt nạ màu sắc
    draw = img.copy() # Sao chép ảnh màu tương ứng để vẽ lên
    for bbox in bboxes:
        x, y, w, h = bbox
        # Vẽ khối hộp bao quanh biển báo
        cv2.rectangle(draw, (x,y), (x+w,y+h), (0,255,255), 4) # vẽ hình chữ nhật bao quanh vật
    results.append(draw)"""
"""results = []
mask = filter_signs_by_color(bgr_images) # lọc theo màu sắc
bboxes = get_boxes_from_mask(mask) # tìm kiếm khung bao của các vật từ mặt nạ màu sắc
draw = bgr_images.copy() # Sao chép ảnh màu tương ứng để vẽ lên
for bbox in bboxes:
    x, y, w, h = bbox
    # Vẽ khối hộp bao quanh biển báo
    cv2.rectangle(draw, (x,y), (x+w,y+h), (0,255,255), 4) # vẽ hình chữ nhật bao quanh vật
results.append(draw)
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
model.eval()"""





def detect_traffic_signs(img):
    results = []
    mask = filter_signs_by_color(img)  # lọc theo màu sắc
    bboxes = get_boxes_from_mask(mask)  # tìm kiếm khung bao của các vật từ mặt nạ màu sắc
    draw = img.copy()  # Sao chép ảnh màu tương ứng để vẽ lên
    for bbox in bboxes:
        x, y, w, h = bbox
        # Vẽ khối hộp bao quanh biển báo
        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 255), 4)  # vẽ hình chữ nhật bao quanh vật
    results.append(draw)
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




    """Phát hiện biển báo
    """

    # Các lớp biển báo
    classes = pd.read_csv('./Data/labels.csv')

    # Phát hiện biển báo theo màu sắc
    mask = filter_signs_by_color(img)
    bboxes = get_boxes_from_mask(mask)

    # Tiền xử lý
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0

    # Phân loại biển báo dùng CNN
    signs = []
    for bbox in bboxes:
        # Cắt vùng cần phân loại
        x, y, w, h = bbox
        sub_image = img[y:y+h, x:x+w]

        if sub_image.shape[0] < 20 or sub_image.shape[1] < 20:
            continue

        # Tiền xử lý
        sub_image = cv2.resize(sub_image, (32, 32))
        sub_image = np.expand_dims(sub_image, axis=0)

        # Sử dụng CNN để phân loại biển báo
        sub_image_tensor = torch.from_numpy(sub_image).permute(0, 3, 1, 2)
        with torch.no_grad():
            preds = model(sub_image_tensor)
        preds = preds[0].numpy()
        cls = np.argmax(preds)
        score = preds[cls]

        # Loại bỏ các vật không phải biển báo - thuộc lớp unknown
        if cls == 0:
            continue

        # Loại bỏ các vật có độ tin cậy thấp
        if score < 0.9:
            continue
        label_name = classes.loc[classes['ClassId'] == cls, 'Name'].values[0]
        signs.append([label_name, x, y, w, h])
        text_lable = []
        # Vẽ các kết quả
        if draw is not None:
            text = str(cls)+" "+label_name + ' ' +str(round(score, 2))
            text_lable.append(text)
            cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 255), 4)
            # Adjust font size for labels
            # label
            """font_scale = 0.25
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(draw, (x, y - text_height - baseline), (x + text_width, y), (0, 255, 0), cv2.FILLED)
            cv2.putText(draw, text, (x, y - baseline), font, font_scale, (0, 0, 0), thickness)"""

    return signs , text_lable , results

"""results = []
draw = bgr_images.copy()
signs = detect_traffic_signs(bgr_images,)
results.append(draw)"""




"""for i, img in enumerate(bgr_images):
    draw = img.copy()
    signs = detect_traffic_signs(img, model, draw=draw)
    results.append(draw)"""


def show_image(image):
    """Display a single image."""
    for i , j in enumerate(image):
        plt.figure(figsize=(8, 6))  # Set figure size
        plt.imshow(cv2.cvtColor(j, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

#show_image(results)


def load_img (img ):
    #img = cv2.imread(img)
    signs, text, results = detect_traffic_signs(img)
    draw = img.copy()
    return results , text