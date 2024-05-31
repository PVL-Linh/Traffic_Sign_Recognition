"""
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image
from traffic_sign_detection import load_img

# Khởi tạo một cửa sổ Tkinter
root = Tk()
root.withdraw()  # Ẩn cửa sổ chính

# Hiển thị hộp thoại chọn tệp tin
filename = askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
img = cv2.imread(filename)
img , label = load_img(img)
image = Image.fromarray(img[0])
print(label)
image.show()
#load_img(img)
#cv2.imshow(image)
#cv2.waitKey(0)  # Đợi cho đến khi người dùng đóng cửa sổ
#cv2.destroyAllWindows()
"""

"""
from tkinter.filedialog import askopenfilename
import cv2
from traffic_sign_detection import load_img
from tkinter import Tk, filedialog
from tkinter import ttk
from PIL import ImageTk, Image
import tkinter
def open_image():
    # Mở hộp thoại chọn file ảnh
    #file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    filename = askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    img = cv2.imread(filename)
    img , label = load_img(img)
    image = Image.fromarray(img[0])
    print(label)
    #image.show()
    # Nếu người dùng đã chọn file ảnh
    if filename:
        # Đọc và hiển thị ảnh
        image = image.resize((500, 500))  # Thay đổi kích thước ảnh
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo  # Lưu tham chiếu đến ảnh để tránh hiện tượng bị mất ảnh

# Tạo cửa sổ chính
root = Tk()
root.title("Image Selector")

# Tạo tab
tab_control = ttk.Notebook(root)
tab = ttk.Frame(tab_control)
tab_control.add(tab, text="Image Viewer")
tab_control.pack(expand=True, fill="both")

# Tạo canvas để hiển thị khung hình vuông
canvas = tkinter.Canvas(tab, width=500, height=500)
canvas.pack()
canvas.create_rectangle(0, 0, 500, 500, outline="black")

# Tạo nút chọn ảnh
select_button = ttk.Button(tab, text="Chọn ảnh", command=open_image)
select_button.pack()

# Khởi chạy giao diện
root.mainloop()

"""
from tkinter.filedialog import askopenfilename
import cv2
from traffic_sign_detection_1 import load_img
from tkinter import Tk, filedialog
from tkinter import ttk
from PIL import ImageTk, Image
import tkinter
import subprocess

def open_image():
    # Mở hộp thoại chọn file ảnh
    filename = askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    img = cv2.imread(filename)
    img, label = load_img(img)
    image = Image.fromarray(img[0])
    print(label)
    # image.show()
    # Nếu người dùng đã chọn file ảnh
    if filename:
        # Đọc và hiển thị ảnh
        image = image.resize((500, 500))  # Thay đổi kích thước ảnh
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo  # Lưu tham chiếu đến ảnh để tránh hiện tượng bị mất ảnh

        # Hiển thị văn bản bên phải ảnh
        text_label = label
        a = 150
        for i in text_label:
            a += 100
            text = "Nhãn: " + str(i)
            canvas.create_text(520, a, anchor="nw", text=text, font=("Arial", 14))


import subprocess

import cv2
import subprocess

def play_video():
    # Mở hộp thoại chọn file video
    filename = askopenfilename(filetypes=[("Video Files", "*.mp4")])

    if filename:
        cap = cv2.VideoCapture(filename)

        # Load pre-trained Haar cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Chuyển đổi frame sang ảnh xám để phát hiện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Phát hiện khuôn mặt trong frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Vẽ hình chữ nhật xung quanh khuôn mặt đã phát hiện
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Hiển thị frame trong cửa sổ video
            cv2.imshow("Video", frame)

            # Thoát khỏi vòng lặp khi nhấn phím "q"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Chạy lệnh trong terminal khi nhấn vào nút
            if cv2.waitKey(1) & 0xFF == ord('n'):
                command = "your_command_here"  # Thay "your_command_here" bằng lệnh bạn muốn chạy
                subprocess.Popen(command, shell=True)

        # Giải phóng các tài nguyên
        cap.release()
        cv2.destroyAllWindows()


root = Tk()
root.title("Biển báo giao thông")

# Tạo tab
tab_control = ttk.Notebook(root)
tab = ttk.Frame(tab_control)

# Thêm label vào tab và đặt thuộc tính font cho label
label = ttk.Label(tab, text="Nhận diện biển báo giao thông", font=("Arial", 14))
label.pack(pady=30)

tab_control.add(tab, text="Nhận diện biển báo giao thông")

# Đặt khung ở giữa bằng cách sử dụng phương thức "pack"
tab_control.pack(expand=True, fill='both')

# Tạo canvas để hiển thị khung hình vuông
canvas = tkinter.Canvas(tab, width=850, height=500)
canvas.pack()

# Tạo nút chọn ảnh
select_button = ttk.Button(tab, text="Chọn ảnh", command=open_image)
select_button.place(relx=0.4, y=70, anchor="center")  # Đặt vị trí của nút trên canvas

# Tạo nút hiển thị video
video_button = ttk.Button(tab, text="Chọn video", command=play_video)
video_button.place(relx=0.6, y=70, anchor="center")  # Đặt vị trí của nút trên canvas

# Khởi chạy giao diện
root.mainloop()