import time
import pickle
import torch
import socket
from multiprocessing import Process, Queue
from Dataset import generate_car_data, generate_forklift_data, generate_uav_data
from Model import LSTM
import run

# 定义全局参数
INPUT_DIM = 10
TRAIN_EPOCHS = 1500
SFT_EPOCHS = 300
PER_POSITIVE = 0.8
NUM_SAMPLES_TRAIN = 5000
NUM_SAMPLES_TEST = 500
NUM_SAMPLES_FINETUNE = 1000
INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 3
PRED_OUTPUT_SIZE = 3
CLAS_OUTPUT_SIZE = 5





def download_data(NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 生成训练集
    forklist_train_x, forklift_train_y, _ = generate_forklift_data(num_samples=NUM_SAMPLES_TRAIN, input_dim=INPUT_DIM, per_positive=PER_POSITIVE)
    torch.save(forklist_train_x, 'Dataset/traindataset.pt')
    torch.save(forklift_train_y, 'Dataset/trainlabels.pt')

    # 生成测试集
    forklift_test_x, forklift_test_y, _ = generate_forklift_data(num_samples=NUM_SAMPLES_TEST, input_dim=INPUT_DIM, per_positive=PER_POSITIVE)
    torch.save(forklist_train_x, 'Dataset/testdataset.pt')
    torch.save(forklift_train_y, 'Dataset/testlabels.pt')

    # 生成微调
    forklift_finetune_x, forklift_finetune_y, forklift_device_finetune_list = generate_forklift_data(num_samples=NUM_SAMPLES_FINETUNE, input_dim=INPUT_DIM, per_positive=PER_POSITIVE)
    torch.save(forklist_train_x, 'Dataset/finetunedataset.pt')
    torch.save(forklift_train_y, 'Dataset/finetunelabels.pt')

def main():
     # 初始化 LSTM 模型
    car_lstm = run.load_model()
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"云端：使用设备 {device}")
    car_lstm = car_lstm.to(device)
    lstm = lstm.to(device)

    # 加载数据集
    forklist_train_x = torch.load('Dataset/traindataset.pt')
    forklift_train_y =  torch.load('Dataset/trainlabels.pt')
    forklist_train_x, forklift_train_y = forklist_train_x.to(device), forklift_train_y.to(device)
    forklift_test_x = torch.load('Dataset/testdataset.pt')
    forklift_test_y =  torch.load('Dataset/testlabels.pt')
    forklift_test_x, forklift_test_y = forklift_test_x.to(device), forklift_test_y.to(device)
    forklift_finetune_x = torch.load('Dataset/finetunedataset.pt')
    forklift_finetune_y = torch.load('Dataset/finetunelabels.pt')
    forklift_finetune_x, forklift_finetune_y = forklift_finetune_x.to(device), forklift_finetune_y.to(device)

    # 智能叉车模型训练
    start_time = time.time()
    forklift_lstm, train_loss_pos_list, train_accuracy_list, train_loss_list, train_epoch_list = run.train_model(
        lstm, forklist_train_x, forklift_train_y, TRAIN_EPOCHS)
    end_time = time.time()
    test_loss_forklift1, test_acc_forklift1, _, _, _ = run.test_model(forklift_lstm, forklift_test_x, forklift_test_y)
    train_time = end_time-start_time
    
    # 小车模型直接用于叉车
    test_loss_forklift2, test_acc_forklift2, _, _, _ = run.test_model(car_lstm, forklift_test_x, forklift_test_y)

    # 智能叉车微调
    start_time = time.time()
    forklift_finetune_lstm, fine_tune_loss_list, fine_tune_accuracy_list, _, fine_tune_epoch_list = run.train_model(
        car_lstm, forklift_finetune_x, forklift_finetune_y, SFT_EPOCHS)
    end_time = time.time()
    test_loss_forklift3, test_acc_forklift3, _, _, _ = run.test_model(forklift_finetune_lstm, forklift_test_x, forklift_test_y)
    finetune_time = end_time-start_time
  

    print(f"小车模型训练完成, 准确率: {test_acc_forklift1}, 损失值：{test_loss_forklift1}, 训练时间：{train_time}")
    print(f"小车模型直接用于叉车, 准确率: {test_acc_forklift2}, 损失值：{test_loss_forklift2}")
    print(f"模型微调完成, 准确率: {test_acc_forklift3}, 损失值：{test_loss_forklift3}, 训练时间：{finetune_time}")

if __name__ == "__main__":
    download_data(NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE)
    main()