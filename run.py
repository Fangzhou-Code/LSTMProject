import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Model import LSTM  
import hashlib
import hmac
import uuid
import rsa 
import numpy as np
import random

# 初始化模型和数据
def initialize_model_and_data():
    INPUT_SIZE = 3
    HIDDEN_SIZE = 32
    NUM_LAYERS = 3
    OUTPUT_SIZE = 1

    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

    train_x = torch.load('Dataset/data1.mat')
    train_y = torch.load('Dataset/label1.mat')

    return lstm, train_x, train_y

# 训练模型
def train_model(lstm, train_x, train_y):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=1e-2)

    max_epochs = 100 # 训练轮次
    loss_list = []
    epoch_list = []
    accuracy_list = []

    for epoch in range(max_epochs):
        output = lstm(train_x)
        loss = loss_function(output, train_y)
        acc = 0

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        threshold = 0.5
        predicted_labels = (output > threshold).int()
        correct_predictions = (predicted_labels == train_y)
        acc = correct_predictions.sum().float() / train_y.size(0) * 100

        if loss.item() < 1e-5:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        epoch_list.append(epoch + 1)
        loss_list.append(loss.item())
        accuracy_list.append(acc)

    return  accuracy_list, loss_list, epoch_list

# 保存模型
def save_model(lstm):
    torch.save(lstm, 'Model/model1.mat')

# 加载模型
def load_model():
    return torch.load('Model/model1.mat')

# 测试模型
def test_model(lstm, test_x, test_y):
    predicted_y = lstm(test_x)

    loss_function = nn.MSELoss()
    loss = loss_function(predicted_y, test_y)

    threshold = 0.5
    predicted_labels = (predicted_y > threshold).int()
    correct_predictions = (predicted_labels == test_y)
    acc = correct_predictions.sum().float() / test_y.size(0) * 100

    return loss.item(), acc

# 画图
def plot_curve(accuracy_list, loss_list, epoch_list):
    plt.figure(figsize=(12, 6))
    
    # 绘制损失图
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, loss_list, label='Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率图
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, accuracy_list, label='Accuracy', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


class DeviceAuthentication:
    def __init__(self, device_id, manufacturer):
        self.device_id = device_id
        self.manufacturer = manufacturer
        self.identifier = self.device_id+self.manufacturer
    def authenticate_device(self):
        # 假设这是一个简单的身份验证逻辑，定义小车的行为模式
        return True
    def issue_credentials(self):
        # 生成随机的凭证和密钥对
        credential = str(uuid.uuid4())
        public_key, private_key = self.generate_key_pair()
        return credential, public_key, private_key
    def generate_key_pair(self):
        # 将字符串转换为唯一数字
        unique_number = self.string_to_unique_number(self.identifier)
        # 生成RSA公私钥对
        pubkey, privkey = rsa.newkeys(2048) 
         # 将公私钥对保存到文件中
        with open('key_pair/public_key.pem', 'wb') as public_key_file:
            public_key_file.write(pubkey.save_pkcs1())
        with open('key_pair/private_key.pem', 'wb') as private_key_file:
            private_key_file.write(privkey.save_pkcs1())
        return pubkey.save_pkcs1(), privkey.save_pkcs1()
    def load_keys(self):
        # 加载已保存的公私钥对
        with open('key_pair/public_key.pem', 'rb') as public_key_file:
            pubkey = rsa.PublicKey.load_pkcs1(public_key_file.read())
        with open('key_pair/private_key.pem', 'rb') as private_key_file:
            privkey = rsa.PrivateKey.load_pkcs1(private_key_file.read())
        return pubkey, privkey
    def string_to_unique_number(self, s): 
        # 使用 SHA-256 哈希函数
        hash_object = hashlib.sha256(s.encode())
        # 以 16 进制格式返回哈希值
        hex_dig = hash_object.hexdigest()
        # 分割十六进制字符串
        parts = [hex_dig[i:i+8] for i in range(0, len(hex_dig), 8)]
        # 转换每个子字符串为整数并相加
        unique_number = sum(int(part, 16) for part in parts)
        return unique_number

if __name__ == "__main__":
    start_time = time.time() 

    # 训练
    lstm, train_x, train_y = initialize_model_and_data()
    accuracy_list, loss_list, epoch_list = train_model(lstm, train_x, train_y)
    save_model(lstm)
    print("...Training Finished...")

    # 测试
    test_x = torch.load('Dataset/data2.mat')
    test_y = torch.load('Dataset/label2.mat')
    lstm = load_model()
    test_loss, test_acc = test_model(lstm, test_x, test_y)
    end_time = time.time()

    # 输出
    print('Test Loss: {:.5f}'.format(test_loss))
    print('Test Accuracy: {:.2f}%'.format(test_acc))
    print("...Test Finished...")
    execution_time = end_time - start_time
    print(f"模型运行时间: {execution_time}秒")

    # 画图
    plot_curve(accuracy_list, loss_list, epoch_list)

    # 模拟一个设备
    device_id = "device123"
    manufacturer = "Example Inc."

    # 实例化设备验证类
    device = DeviceAuthentication(device_id, manufacturer)

    # 进行设备身份验证
    if device.authenticate_device():
        # 如果设备验证通过，下发凭证和密钥对
        credential, public_key, private_key = device.issue_credentials()
        print("Credential:", credential)
        print("Public Key:", public_key)
        print("Private Key:", private_key)
    else:
        print("Device authentication failed.")
