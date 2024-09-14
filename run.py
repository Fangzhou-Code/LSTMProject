'''
注意：这个文件主函数里面的代码存在很多问题，对模型的训练和测试请参考train_test_model.py
'''

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
from matplotlib.ticker import PercentFormatter
import pickle

# 初始化模型和数据
def initialize_model_data(INPUT_SIZE, input_dim, per_positive, NEW_DATA = 0):
    INPUT_SIZE = INPUT_SIZE
    HIDDEN_SIZE = 64
    NUM_LAYERS = 3
    PRED_OUTPUT_SIZE = 3
    CLAS_OUTPUT_SIZE = 5
    if NEW_DATA:
        train_x, train_y,device_list = Dataset.generate_car_data(num_samples=10000, input_dim=input_dim, per_positive=per_positive)
        torch.save(train_x, 'Dataset/traindataset.pt')
        torch.save(train_y, 'Dataset/trainlabels.pt')
        # 保存list数组到本地文件
        with open('Dataset/traindevicelist.pkl', 'wb') as f:
            pickle.dump(device_list, f)
        print("...train Create Finished...")
    else:
        train_x = torch.load('Dataset/traindataset.pt')
        train_y = torch.load('Dataset/trainlabels.pt')
        with open('Dataset/traindevicelist.pkl', 'rb') as f:
            device_list = pickle.load(f)
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)
    return lstm, train_x, train_y, device_list
def initialize_test_data(input_dim, per_positive, NEW_DATA= 0):
    if NEW_DATA:
        test_x, test_y,device_list = Dataset.generate_car_data(num_samples=1000, input_dim=input_dim, per_positive=per_positive)
        torch.save(test_x, 'Dataset/testdataset.pt')
        torch.save(test_y, 'Dataset/testlabels.pt') 
        # 保存list数组到本地文件
        with open('Dataset/testdevicelist.pkl', 'wb') as f:
            pickle.dump(device_list, f)
        print("...test Create Finished...")
    else:
        test_x = torch.load('Dataset/testdataset.pt')
        test_y = torch.load('Dataset/testlabels.pt')
        with open('Dataset/testdevicelist.pkl', 'rb') as f:
            device_list = pickle.load(f)
    return test_x, test_y, device_list

# 训练模型
def train_model(lstm, train_x, train_y, max_epochs, lr=1e-2, weight_decay=1e-5):
    optimizer = optim.Adam(lstm.parameters(), lr=lr, weight_decay=weight_decay)

    train_y_pred = train_x[:, -1, :]  # 预测任务标签
    train_y_clas = train_y  # 分类任务标签

    train_x = train_x[:, :-1, :]
    
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
            loss_pred, accuracy, _, _, _ = test_model(lstm, train_x, train_y)
            epoch_list.append(epoch + 1)
            loss_list.append(loss.item())
            loss_pred_list.append(loss_pred.detach().cpu().numpy())
            accuracy_list.append(accuracy.detach().cpu().numpy())
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    return lstm, loss_pred_list, accuracy_list, loss_list, epoch_list


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
    accuracy = torch.mean(torch.eq(pred_labels, test_y_clas).all(dim=1).float()) 

    return loss_pos, accuracy, pred_y_pred, pred_y_clas, pred_labels



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

def plot_comparison(test_loss_edge, test_acc_edge, test_loss_cloud, test_acc_cloud):
    # 如果输入是 GPU 张量，先移动到 CPU 并转换为 numpy 数组
    if isinstance(test_loss_edge, torch.Tensor):
        test_loss_edge = test_loss_edge.detach().cpu().numpy()
    if isinstance(test_acc_edge, torch.Tensor):
        test_acc_edge = test_acc_edge.detach().cpu().numpy()
    if isinstance(test_loss_cloud, torch.Tensor):
        test_loss_cloud = test_loss_cloud.detach().cpu().numpy()
    if isinstance(test_acc_cloud, torch.Tensor):
        test_acc_cloud = test_acc_cloud.detach().cpu().numpy()

    # 设置柱状图数据
    labels = ['Before fine-tuning', 'After fine-tuning']
    losses = [test_loss_edge, test_loss_cloud]
    accuracies = [test_acc_edge, test_acc_cloud]

    x = np.arange(len(labels))  # 横轴刻度的位置
    width = 0.35  # 柱状图的宽度

    # 创建图形和子图
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制损失的柱状图
    ax1.bar(x - width/2, losses, width, label='Loss', color='#3CB371')
    ax1.set_xlabel('Before fine-tuning VS. After fine-tuning')
    ax1.set_ylabel('Loss', color='#3CB371')
    ax1.tick_params(axis='y', labelcolor='#3CB371')
    ax1.legend(loc='upper left')
    # ax1.set_ylim(50, 70)  # 限制损失的 y 轴范围

    # 在同一个子图中共享 x 轴绘制准确率的柱状图
    ax2 = ax1.twinx()  # 使用相同的 x 轴
    ax2.bar(x + width/2, accuracies, width, label='Accuracy', color='#FF8C00')
    ax2.set_ylabel('Accuracy', color='#FF8C00')
    ax2.tick_params(axis='y', labelcolor='#FF8C00')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0.5, 1)  # 限制准确率的 y 轴范围

    # 添加标题和标签
    # plt.title('Comparison of Test Results between Edge and Cloud Deployment')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    # 自动调整布局
    fig.tight_layout()

    # 展示图像
    plt.show()

class DeviceAuthentication:
    def __init__(self, device_id, manufacturer):
        self.device_id = device_id
        self.manufacturer = manufacturer
        self.identifier = self.device_id+self.manufacturer
    def authenticate_device(self, pred_labels): #[1x4] 1000 0100 0010 0001
        for i in range(pred_labels.size(0)):
            single_pred_label = pred_labels[i]
            # 将 one-hot 编码的标签转换为类别索引
            label_index = torch.argmax(single_pred_label).item()
            # print(f"Sample {i} prediction: {single_pred_label}, Label index: {label_index}")
            if label_index == 0 or label_index == 1:
                # 如果设备验证通过，下发凭证和密钥对
                print("Credential")
                # credential, public_key, private_key = device.issue_credentials() 
                # print("Credential:", credential)
                # print("Public Key:", public_key)
                # print("Private Key:", private_key)
            else:
                print("Device authentication failed.")
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

# 生成小车的最后使用上次身份凭证的时间
def generate_final_time(num, low, high):
    sum = 0
    for _ in range(num):
        random_num = random.randint(low, high)
        sum += random_num
    return sum

if __name__ == "__main__":
    start_time = time.time() 

    # 初始化
    input_dim = 20
    lstm, train_x, train_y,device_train = initialize_model_data(3, input_dim)

    # 加速（CPU记得注释掉）
    lstm = lstm.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    

    # 训练
    loss_pos_list, accuracy_list, loss_list, epoch_list = train_model(lstm, train_x, 
                                                                      train_y, max_epochs=5000)
    save_model(lstm)
    print("...Training Finished...")    
    end_time = time.time()
    
    # 测试
    lstm = load_model()
    per_positive = 0.6 # 初始化小车的正样本概率
    max_num = 10 # 身份凭证最大使用次数
    use_num = 0 # 初始化使用次数
    car_num = 100 # 验证小车数量
    test_loss_sum = 0 # 小车更新身份凭证loss总和
    test_acc_sum = 0 # 小车更新身份凭证acc总和
    ori_loss_sum = 0 # 小车不更新身份凭证loss总和
    ori_acc_sum = 0 # 小车不更新身份凭证acc总和
    # 进行设备身份验证，发送身份凭证
    
    for i in range(car_num): # 每一辆小车进行验证
        # 模拟一个小车 （后续可以随机或者导入数据集）
        device_id = "device123" 
        manufacturer = "Example Inc."
        device = DeviceAuthentication(device_id, manufacturer)# 实例化设备验证类
        final_time = generate_final_time(10, 0, 3) + input_dim # 时间必须大于input_dim
        test_x, test_y,device_test = initialize_test_data(final_time) #[1,final_time,3] 
        test_x = test_x.cuda() 
        test_y = test_y.cuda()
        test_loss_list = []
        test_acc_list = []
        for i in range(input_dim, final_time):
            test_loss, test_acc, pred_labels = test_model(lstm, test_x[:, i-input_dim:i, :], test_y)
            # test_loss_list.append(test_loss.detach().cpu())
            # test_acc_list.append(test_acc.detach().cpu())
            if i==20: # 存储不更新身份凭证下小车的acc和loss
                ori_loss_sum += test_loss.detach().cpu()
                ori_acc_sum += test_acc.detach().cpu()
            if device.authenticate_device(pred_labels):
                # 如果设备验证通过，下发凭证和密钥对
                print("Credential")
                # credential, public_key, private_key = device.issue_credentials() 
                # print("Credential:", credential)
                # print("Public Key:", public_key)
                # print("Private Key:", private_key)
            else:
                print("Device authentication failed.")
                break
        print("Maximum usage reached, destroy authentication.")
        print("test_loss_list",type(test_loss_list))
        # test_loss_sum += np.mean(test_loss_list)
        # test_acc_sum += np.mean(test_acc_list)
        test_loss_sum += test_loss.detach().cpu()
        test_acc_sum += test_acc.detach().cpu()
    # 输出
    print('不更新身份凭证Test Loss: {:.5f}'.format(ori_loss_sum))
    print('不更新身份凭证Test Accuracy: {:.2f}%'.format(ori_acc_sum))
    print('更新身份凭证Test Loss: {:.5f}'.format(test_loss_sum))
    print('更新身份凭证Test Accuracy: {:.2f}%'.format(test_acc_sum)) 
    print("...Test Finished...")
    execution_time = end_time - start_time
    print(f"模型运行时间: {execution_time}秒")

    # 画图
    plot_curve(loss_pos_list, accuracy_list, loss_list, epoch_list) # 训练集acc, loss
    plt.savefig('./results/training_curve.pdf')
    # 柱状图数据
    categories = ['Original', 'LSTM不更新身份凭证', 'LSTM更新身份凭证']
    # 将错误率和错误率与测试准确率的乘积转换为百分比
    values = [per_positive * 100, (ori_acc_sum / 100) * 100, (test_acc_sum / 100) * 100,]
    # 绘制柱状图
    plt.bar(categories, values, color=['#FFA07A', '#87CEEB', '#4682B4'])
    plt.ylabel('Percentage')  # 纵坐标标签改为百分比
    # 设置中文字体
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(u'设备身份验证的准确率')
    plt.ylim(0, 100)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())  # 设置纵坐标格式为百分比
    plt.show()
    plt.savefig('./results/detect_curve.pdf')