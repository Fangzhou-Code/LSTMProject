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
TRAIN_EPOCHS = 500
SFT_EPOCHS = 500
PER_POSITIVE = 0.6
NUM_SAMPLES_TRAIN = 5000
NUM_SAMPLES_TEST = 500
NUM_SAMPLES_FINETUNE = 100
INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 3
PRED_OUTPUT_SIZE = 3
CLAS_OUTPUT_SIZE = 5

def save_data(train_x, train_y, device_list, prefix='train'):
    """保存数据集及设备列表到指定路径"""
    try:
        torch.save(train_x, f'Dataset/{prefix}dataset.pt')
        torch.save(train_y, f'Dataset/{prefix}labels.pt')
        with open(f'Dataset/{prefix}devicelist.pkl', 'wb') as f:
            pickle.dump(device_list, f)
        print(f"...{prefix} dataset Create Finished...")
    except Exception as e:
        print(f"Error saving {prefix} data: {e}")

# 主函数，创建三个进程
def main():
     # 初始化 LSTM 模型
    car_lstm = run.load_model()
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"云端：使用设备 {device}")
    car_lstm = car_lstm.to(device)
    lstm = lstm.to(device)

   
    # 智能叉车模型训练
    forklist_train_x, forklift_train_y, forklift_device_train_list = generate_forklift_data(num_samples=NUM_SAMPLES_TRAIN, input_dim=INPUT_DIM, per_positive=PER_POSITIVE)
    forklist_train_x, forklift_train_y = forklist_train_x.to(device), forklift_train_y.to(device)
    # save_data(forklist_train_x, forklist_train_x, forklift_device_train_list, prefix='train')
    start_time = time.time()
    forklift_lstm, train_loss_pos_list, train_accuracy_list, train_loss_list, train_epoch_list = run.train_model(
        lstm, forklist_train_x, forklift_train_y, TRAIN_EPOCHS)
    end_time = time.time()
    # run.save_model(forklift_lstm)
    forklift_test_x, forklift_test_y, forklift_device_test_list = generate_forklift_data(num_samples=NUM_SAMPLES_TEST, input_dim=INPUT_DIM, per_positive=PER_POSITIVE)
    forklift_test_x, forklift_test_y = forklift_test_x.to(device), forklift_test_y.to(device)
    # save_data(forklift_test_x, forklift_test_y, forklift_device_test_list, prefix='test')
    test_loss_forklift1, test_acc_forklift1, _, _, _ = run.test_model(forklift_lstm, forklift_test_x, forklift_test_y)
    train_time = end_time-start_time
    
    
    # 小车模型直接用于叉车
    test_loss_forklift2, test_acc_forklift2, _, _, _ = run.test_model(car_lstm, forklift_test_x, forklift_test_y)
    print(f"小车模型直接用于叉车, 准确率: {test_acc_forklift2}, 损失值：{test_loss_forklift2}")

    # 智能叉车微调
    forklift_finetune_x, forklift_finetune_y, forklift_device_finetune_list = generate_forklift_data(num_samples=NUM_SAMPLES_FINETUNE, input_dim=INPUT_DIM, per_positive=PER_POSITIVE)
    forklift_finetune_x, forklift_finetune_y = forklift_finetune_x.to(device), forklift_finetune_y.to(device)
    # save_data(forklift_finetune_x, forklift_finetune_y, forklift_device_finetune_list, prefix='finetune')
    start_time = time.time()
    forklift_finetune_lstm, fine_tune_loss_list, fine_tune_accuracy_list, _, fine_tune_epoch_list = run.train_model(
        car_lstm, forklift_finetune_x, forklift_finetune_y, SFT_EPOCHS)
    end_time = time.time()
    # run.save_model(forklift_finetune_lstm)
    test_loss_forklift3, test_acc_forklift3, _, _, _ = run.test_model(forklift_finetune_lstm, forklift_test_x, forklift_test_y)
    finetune_time = end_time-start_time
  

    print(f"小车模型训练完成, 准确率: {test_acc_forklift1}, 损失值：{test_loss_forklift1}, 训练时间：{train_time}")
    print(f"小车模型直接用于叉车, 准确率: {test_acc_forklift2}, 损失值：{test_loss_forklift2}")
    print(f"模型微调完成, 准确率: {test_acc_forklift3}, 损失值：{test_loss_forklift3}, 训练时间：{finetune_time}")

if __name__ == "__main__":
    main()
