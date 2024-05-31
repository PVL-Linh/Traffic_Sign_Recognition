"""import gc, os, cv2, PIL, torch
import torchvision as tv
import torch.nn as nn
import torchsummary as ts
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

labels_df = pd.read_csv('./Data/labels.csv')
print(labels_df)



x , y = [] , []  # X to store images and y to store respective labels
data_dir = './Data/myData'
for folder in range(43):
    folder_path = os.path.join(data_dir,str(folder)) # os.path.join just join both string
    for i,img in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path,img)
        # PIL load the image as PIL object and ToTensor() convert this to a Tensor
        img_tensor = tv.transforms.ToTensor()(PIL.Image.open(img_path))
        x.append(img_tensor.tolist()) # convert the tensor to list of list and append
        y.append(folder)
    print('folder of label',folder,'images loaded. Number of samples :',i+1)
x = np.array(x)
y = np.array(y)


np.unique(y,return_counts=True)


x = x.reshape(x.shape[0],3*32*32) # flatten x as RandomOverSampler only accepts 2-D matrix
# RandomOverSampler method duplicates samples in the minority class to balance dataset
x,y = RandomOverSampler().fit_resample(x,y)
x = x.reshape(x.shape[0],3,32,32) # reshaped again as it was
print(x.shape)
print(y.shape)

np.unique(y,return_counts=True)



# Stratified split on the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,stratify=y)
del x,y
gc.collect() # delete x,y and free the memory
print ('xtrain ',xtrain.shape,'\nxtest', xtest.shape,'\nytrain', ytrain.shape,'\nytest', ytest.shape) # splited data shapes



#plt.figure(figsize=(20,20))
# make_grid creates a grid of 100 images and show it
#plt.imshow(tv.utils.make_grid(torch.tensor(xtrain[:100]),nrow=10).permute(1,2,0))
#plt.axis('off') # To remove xticks and yticks
#plt.show()
#print('\n\nLabels of the above images :\n')
#print(ytrain[:100])




xtrain = torch.from_numpy(xtrain)
ytrain = torch.from_numpy(ytrain)
xtest = torch.from_numpy(xtest)
ytest = torch.from_numpy(ytest)

model = nn.Sequential(
    # 1st convolutional network Layers
    nn.Conv2d(3, 16, (2, 2), (1, 1), 'same'),  # Convolution
    nn.BatchNorm2d(16),  # Normalization
    nn.ReLU(True),  # Activation
    nn.MaxPool2d((2, 2)),  # Pooling

    # 2nd convolutional network Layers
    nn.Conv2d(16, 32, (2, 2), (1, 1), 'same'),  # Convolution
    nn.BatchNorm2d(32),  # Normalization
    nn.ReLU(True),  # Activation
    nn.MaxPool2d((2, 2)),  # Pooling

    # 3rd convolutional network Layers
    nn.Conv2d(32, 64, (2, 2), (1, 1), 'same'),  # Convolution
    nn.BatchNorm2d(64),  # Normalization
    nn.ReLU(True),  # Activation
    nn.MaxPool2d((2, 2)),  # Pooling

    # Flatten Data
    nn.Flatten(),  # Flatten

    # feed forward Layers
    nn.Linear(1024, 256),  # Linear
    nn.ReLU(True),  # Activation
    nn.Linear(256, 43)  # Linear
)

# Send model to Cuda Memory
model = model.to(torch.device('cuda'), non_blocking=True)
# For Model Summary
ts.summary(model, (3, 32, 32))

def evaluate(model, data, target):
    # sending data and target to cuda memory
    data = data.to(torch.device('cuda'),non_blocking=True)
    target = target.to(torch.device('cuda'),non_blocking=True)
    length = len(target)
    yhat = model(data) # predict on data
    ypred = yhat.argmax(axis=1) # claculate the prediction labels from yhat
    loss = float(nn.functional.cross_entropy(yhat, target)) # calculate the loss
    acc = float((ypred == target).sum() / length) # Calculate accuracy
    print('Loss :',round(loss,4),'- Accuracy :',round(acc,4)) # Print loss and Accuracy
    del data,target,yhat,ypred # delete the used variables
    torch.cuda.empty_cache() # Free the Cuda memory

print('\nInitial Loss and Accuracy on Test Dataset :')
evaluate(model, xtest.float(), ytest)


def train_model(model=model, optimizer=torch.optim.Adam, epochs=5, batch_size=200, steps_per_epochs=200, l2_reg=0,
                max_lr=0.01, grad_clip=0.5):
    hist = [[], [], [], []]  # hist will stores train and test data losses and accuracy of every epochs

    train_ds = [(x, y) for x, y in zip(xtrain, ytrain)]  # Prepare training dataset for Data Loader
    training_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)  # Data Loader used to train model
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size * steps_per_epochs)
    # Data Loader for epoch end evaluation on train data
    del train_ds
    gc.collect()  # Delete the used variable and free up memory

    # Initialized the Optimizer to update weights and bias of model parameters
    optimizer = optimizer(model.parameters(), weight_decay=l2_reg)

    # Initialized the Schedular to update learning rate as per one cycle poicy
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=int(steps_per_epochs * 1.01))

    # Training Started
    for i in range(epochs):

        print('\nEpoch', i + 1, ': [', end="")

        # Load Batches of training data loader
        for j, (xb, yb) in enumerate(training_dl):

            # move the training batch data to cuda memory for faster processing
            xb = xb.to(torch.device('cuda'), non_blocking=True)
            yb = yb.to(torch.device('cuda'), non_blocking=True)

            # Calculate Losses and gradients
            yhat = model(xb.float())
            loss = nn.functional.cross_entropy(yhat, yb)
            loss.backward()

            # Clip the outlier like gradients
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            # Update Weights and bias
            optimizer.step()
            optimizer.zero_grad()

            # Update Learning Rate
            sched.step()

            del xb, yb, yhat
            torch.cuda.empty_cache()
            # delete the used data and free up space

            # print the training epochs progress
            if j % int(steps_per_epochs / 20) == 0:
                print('.', end='')

            # break the loop when all steps of an epoch completed.
            if steps_per_epochs == j:
                break

        # Epochs end evaluation

        device = torch.device('cuda')  # initialized cuda to device

        # load training data batches from train data loader
        for xtrainb, ytrainb in train_dl:
            break

        # move train data to cuda
        xtrain_cuda = xtrainb.to(device, non_blocking=True)
        ytrain_cuda = ytrainb.to(device, non_blocking=True)
        del xtrainb, ytrainb
        gc.collect()
        # delete used variables and free up space

        # Calculate train loss and accuracy
        yhat = model(xtrain_cuda.float())
        ypred = yhat.argmax(axis=1)
        train_loss = float(nn.functional.cross_entropy(yhat, ytrain_cuda))
        train_acc = float((ypred == ytrain_cuda).sum() / len(ytrain_cuda))

        del xtrain_cuda, ytrain_cuda, yhat, ypred
        torch.cuda.empty_cache()
        # delete used variables and free up space

        # move test data to cuda
        xtest_cuda = xtest.to(device, non_blocking=True)
        ytest_cuda = ytest.to(device, non_blocking=True)

        # Calculate test loss and accuracy
        yhat = model(xtest_cuda.float())
        ypred = yhat.argmax(axis=1)
        val_loss = float(nn.functional.cross_entropy(yhat, ytest_cuda))
        val_acc = float((ypred == ytest_cuda).sum() / len(ytest_cuda))

        del xtest_cuda, ytest_cuda, yhat, ypred
        torch.cuda.empty_cache()
        # delete used variables and free up space

        # print the captured train and test loss and accuracy at the end of every epochs
        print('] - Train Loss :', round(train_loss, 4), '- Train Accuracy :', round(train_acc, 4),
              '- Val Loss :', round(val_loss, 4), '- Val Accuracy :', round(val_acc, 4))

        # store that data into the previously blank initialized hist list
        hist[0].append(train_loss)
        hist[1].append(val_loss)
        hist[2].append(train_acc)
        hist[3].append(val_acc)

    # Initialized all the evaluation history of all epochs to a dict
    history = {'Train Loss': hist[0], 'Val Loss': hist[1], 'Train Accuracy': hist[2], 'Val Accuracy': hist[3]}

    # return the history as pandas dataframe
    return pd.DataFrame(history)


history = train_model(model,optimizer=torch.optim.Adam,epochs=25,steps_per_epochs=200,l2_reg=0,max_lr=0.015,grad_clip=0.5)


history.save('traffic_sign_model.h5')
#model = keras.models.load_model("traffic_sign_model_traffic_2.h5")

# Kiểm tra model với dữ liệu mới
print(model.evaluate(xtest, ytest))

"""
import gc
import os
import cv2
import PIL
import torch
import torchvision as tv
import torch.nn as nn
import torchsummary as ts
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# Load dữ liệu
labels_df = pd.read_csv('./Data/labels.csv')
print(labels_df)

x, y = [], []  # X chứa hình ảnh và y chứa nhãn tương ứng
data_dir = './Data/myData'
for folder in range(43):
    folder_path = os.path.join(data_dir, str(folder))
    for i, img in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img)
        img_tensor = tv.transforms.ToTensor()(PIL.Image.open(img_path))
        x.append(img_tensor.tolist())
        y.append(folder)
    print('folder of label', folder, 'images loaded. Number of samples:', i + 1)
x = np.array(x)
y = np.array(y)

# Reshape và cân bằng dữ liệu
x = x.reshape(x.shape[0], 3 * 32 * 32)
x, y = RandomOverSampler().fit_resample(x, y)
x = x.reshape(x.shape[0], 3, 32, 32)
print(x.shape)
print(y.shape)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y)
del x, y
gc.collect()
print('xtrain ', xtrain.shape, '\nxtest', xtest.shape, '\nytrain', ytrain.shape, '\nytest', ytest.shape)

# Xây dựng mô hình CNN
model = nn.Sequential(
    # Các layer convolutional
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

    # Lớp Flatten
    nn.Flatten(),

    # Các lớp fully connected
    nn.Linear(1024, 256),
    nn.ReLU(True),
    nn.Linear(256, 43)
)

# Xem tóm tắt của mô hình
ts.summary(model, (3, 32, 32))

# Đánh giá mô hình trên tập kiểm tra
def evaluate(model, data, target):
    length = len(target)
    yhat = model(data)
    ypred = yhat.argmax(axis=1)
    loss = float(nn.functional.cross_entropy(yhat, target.long()))  # Fix applied here
    acc = float((ypred == target).sum() / length)
    print('Loss:', round(loss, 4), '- Accuracy:', round(acc, 4))

print('\nInitial Loss and Accuracy on Test Dataset:')
evaluate(model, torch.tensor(xtest, dtype=torch.float), torch.tensor(ytest))

# Huấn luyện mô hình
def train_model(model, optimizer=torch.optim.Adam, epochs=5, batch_size=200, steps_per_epochs=200, l2_reg=0,
                max_lr=0.01, grad_clip=0.5):
    hist = [[], [], [], []]

    train_ds = [(x, y) for x, y in zip(xtrain, ytrain)]
    training_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size * steps_per_epochs)
    del train_ds
    gc.collect()

    optimizer = optimizer(model.parameters(), weight_decay=l2_reg)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=int(steps_per_epochs * 1.01))

    for i in range(epochs):
        print('\nEpoch', i + 1, ': [', end="")
        for j, (xb, yb) in enumerate(training_dl):
            yhat = model(xb.float())
            loss = nn.functional.cross_entropy(yhat, yb.long())  # Fix applied here
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            sched.step()
            del xb, yb, yhat
            # print the training epochs progress
            if j % int(steps_per_epochs / 20) == 0:
                print('.', end='')
            # break the loop when all steps of an epoch completed.
            if steps_per_epochs == j:
                break

        device = torch.device('cpu')
        for xtrainb, ytrainb in train_dl:
            break
        xtrain_cuda = xtrainb.to(device)
        ytrain_cuda = ytrainb.to(device)
        del xtrainb, ytrainb
        gc.collect()

        yhat = model(xtrain_cuda.float())
        ypred = yhat.argmax(axis=1)
        train_loss = float(nn.functional.cross_entropy(yhat, ytrain_cuda.long()))  # Fix applied here
        train_acc = float((ypred == ytrain_cuda).sum() / len(ytrain_cuda))
        del xtrain_cuda, ytrain_cuda, yhat, ypred
        # move test data to cpu
        xtest_cuda = torch.tensor(xtest, dtype=torch.float)
        ytest_cuda = torch.tensor(ytest)
        yhat = model(xtest_cuda.float())
        ypred = yhat.argmax(axis=1)
        val_loss = float(nn.functional.cross_entropy(yhat, ytest_cuda.long()))  # Fix applied here
        val_acc = float((ypred == ytest_cuda).sum() / len(ytest_cuda))
        del xtest_cuda, ytest_cuda, yhat, ypred
        print('] - Train Loss:', round(train_loss, 4), '- Train Accuracy:', round(train_acc, 4),
              '- Val Loss:', round(val_loss, 4), '- Val Accuracy:', round(val_acc, 4))
        hist[0].append(train_loss)
        hist[1].append(val_loss)
        hist[2].append(train_acc)
        hist[3].append(val_acc)

    history = {'Train Loss': hist[0], 'Val Loss': hist[1], 'Train Accuracy': hist[2], 'Val Accuracy': hist[3]}
    return pd.DataFrame(history)

history = train_model(model, optimizer=torch.optim.Adam, epochs=25, steps_per_epochs=200, l2_reg=0, max_lr=0.015,
                      grad_clip=0.5)


torch.save(model.state_dict(), 'traffic_sign_model_2.pt')


history.to_csv('traffic_sign_model_history_2.csv')
