import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from Model import LSTM  
import hashlib
import hmac
import uuid
import rsa
import numpy as np
import random
import Dataset


# 初始化模型和数据
def initialize_model_data():
    INPUT_SIZE = 3
    HIDDEN_SIZE = 64
    NUM_LAYERS = 3
    PRED_OUTPUT_SIZE = 3
    CLAS_OUTPUT_SIZE = 4
    train_x, train_y = Dataset.generate_car_data(num_samples=10000, input_dim=20, per_positive=0.7)
    test_x, test_y = Dataset.generate_car_data(num_samples=1000, input_dim=20, per_positive=0.7)
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)
    return lstm, train_x, train_y, test_x, test_y
def initialize_valid_data(input_dim, per_positive):
    valid_x, valid_y = Dataset.generate_car_data(num_samples=1, input_dim=input_dim, per_positive=per_positive)
    return valid_x, valid_y

# 训练模型
def train_model(lstm, train_x, train_y):
    optimizer = optim.Adam(lstm.parameters(), lr=1e-2, weight_decay=1e-5)

    train_y_pred = train_x[:, -1, :]  # 预测任务标签
    train_y_clas = train_y  # 分类任务标签

    train_x = train_x[:, :-1, :]

    max_epochs = 100 # 训练轮次
    
    # 训练轮次
    epoch_list = []
    loss_list = []
    loss_pred_list = []
    accuracy_list = []

    for epoch in range(max_epochs):
        pred_y_pred, pred_y_clas = lstm(train_x)

        loss_pred = lstm.loss_mse(pred_y_pred, train_y_pred)
        loss_ce = lstm.loss_ce(pred_y_clas, train_y_clas)

        loss = loss_pred / 1000 + loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < 1e-5:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            break
        elif (epoch + 1) % 10 == 0:
            # 测试模型在训练集上的预测损失与分类精度用于画图展示
            loss_pred, accuracy, _ = test_model(lstm, train_x, train_y)
            epoch_list.append(epoch + 1)
            loss_list.append(loss.item())
            loss_pred_list.append(loss_pred.detach().cpu().numpy())
            accuracy_list.append(accuracy.detach().cpu().numpy())
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    return loss_pred_list, accuracy_list, loss_list, epoch_list


# 保存模型
def save_model(lstm):
    torch.save(lstm, 'Model/lstmmodel.pt')


# 加载模型
def load_model():
    return torch.load('Model/lstmmodel.pt')


# 测试模型的预测损失与分类精度
def test_model(lstm, test_x, test_y):
    test_y_pred = test_x[:, -1, :]  # 预测任务标签
    test_y_clas = test_y  # 分类任务标签

    test_x = test_x[:, :-1, :]

    pred_y_pred, pred_y_clas = lstm(test_x)

    # 预测任务的loss
    loss_pos = lstm.loss_mse(pred_y_pred, test_y_pred)

    # 分类任务的accuracy
    pred_labels = F.one_hot(torch.argmax(pred_y_clas, dim=1), num_classes=lstm.clas_output_size)
    accuracy = torch.mean(torch.eq(pred_labels, test_y_clas).all(dim=1).float()) * 100

    return loss_pos, accuracy, pred_labels


# 画图
def plot_curve(loss_pred_list, accuracy_list, loss_list, epoch_list):
    plt.figure(figsize=(12, 6))
    
    # 绘制总损失图
    plt.subplot(1, 3, 1)
    plt.plot(epoch_list, loss_list, label='Training Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制预测任务损失图
    plt.subplot(1, 3, 2)
    plt.plot(epoch_list, loss_pred_list, label='Prediction Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制分类任务准确率图
    plt.subplot(1, 3, 3)
    plt.plot(epoch_list, accuracy_list, label='Classification Accuracy', color='green')
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
    def authenticate_device(self, pred_labels):
        if(pred_labels[0,0].item() == 1):
            return True
        else:
            return False
    def issue_credentials(self):
        # 生成随机的凭证和密钥对
        credential = str(uuid.uuid4())
        public_key, private_key = self.generate_key_pair()
        return credential, public_key, private_key
    def generate_key_pair(self):
        # 将字符串转换为唯一数字
        unique_number = self.string_to_unique_number(self.identifier)
        # 生成RSA公私钥对
        pubkey, privkey = rsa.newkeys(2048)  #2048
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

    # 初始化
    lstm, train_x, train_y, test_x, test_y = initialize_model_data()

    # 加速（CPU记得注释掉）
    lstm = lstm.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

    # 训练
    loss_pos_list, accuracy_list, loss_list, epoch_list = train_model(lstm, train_x, train_y)
    save_model(lstm)
    print("...Training Finished...")    
    
    # 测试
    lstm = load_model()
    test_loss, test_acc, pred_labels = test_model(lstm, test_x, test_y)
    end_time = time.time()

    # 输出
    print('Test Loss: {:.5f}'.format(test_loss))
    print('Test Accuracy: {:.2f}%'.format(test_acc))
    print("...Test Finished...")
    execution_time = end_time - start_time
    print(f"模型运行时间: {execution_time}秒")

    per_positive = 0.7 # 初始化小车的正样本概率
    max_num = 10 # 身份凭证最大使用次数
    use_num = 0 # 初始化使用次数
    car_num = test_x.size(0)-1 # 验证小车数量

    # 进行设备身份验证，发送身份凭证
    for i in range(car_num): # 每一辆小车进行验证
        # 模拟一个小车 （后续可以随机或者导入数据集）
        device_id = "device123"
        manufacturer = "Example Inc."
        device = DeviceAuthentication(device_id, manufacturer)# 实例化设备验证类

        while use_num <= 10:
            use_num += 1
            use_time = random.randint(0, 10) # 假设每次申请时间不超过10秒

            # 新的测试集
            valid_x, valid_y = initialize_valid_data(use_time+1, per_positive)
            valid_x = valid_x.cuda() # 加速
            valid_y = valid_y.cuda() # 加速

            _, _, pred_labels = test_model(lstm, valid_x, valid_y) 
            if device.authenticate_device(pred_labels):
                # 如果设备验证通过，下发凭证和密钥对
                credential, public_key, private_key = device.issue_credentials()
                print("Credential:", credential)
                # print("Public Key:", public_key)
                # print("Private Key:", private_key)
            else:
                print("Device authentication failed.")

    # 画图
    plot_curve(loss_pos_list, accuracy_list, loss_list, epoch_list) # 训练集acc, loss
    error_rate = 1- per_positive # 错误率和测试准确率
    # 计算错误率和错误率与测试准确率的乘积
    error_rate_times_test_acc = error_rate * (test_acc.cpu() / 100)
    # 柱状图数据
    categories = ['Original', 'LSTM']
    values = [0.3, error_rate_times_test_acc]
    # 绘制柱状图
    plt.bar(categories, values, color=['red', 'blue'])
    plt.ylabel('Value')
    plt.title('Error sample detection rate')
    plt.show()
