import pandas as pd
import matplotlib.pyplot as plt

# Đọc lịch sử huấn luyện từ tệp CSV
history = pd.read_csv('traffic_sign_model_history.csv')

# Vẽ biểu đồ mất mát trên tập huấn luyện và tập kiểm tra
plt.plot(history['Train Loss'], label='Train Loss')
plt.plot(history['Val Loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Vẽ biểu đồ độ chính xác trên tập huấn luyện và tập kiểm tra
plt.plot(history['Train Accuracy'], label='Train Accuracy')
plt.plot(history['Val Accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
