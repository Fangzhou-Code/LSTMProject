import time, torch, run, json
from Dataset import generate_forklift_data, generate_uav_data
from Model import LSTM

# 定义全局参数
INPUT_DIM = 10
TRAIN_EPOCHS = 1500
SFT_EPOCHS = 300
NUM_SAMPLES_TRAIN = 5000
NUM_SAMPLES_TEST = 500
NUM_SAMPLES_FINETUNE = 1000
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 3
PRED_OUTPUT_SIZE = INPUT_SIZE
CLAS_OUTPUT_SIZE = 5
PER_POSITIVE = 0.6
PER_CONTROL = 0.5
test_num = 10 # 生成测试数据集数量，最后求平均

# 叉车数据集路径
forklift_traindataset_pth = 'Dataset/transferlearning_forklift_traindataset.pt'
forklift_trainlabels_pth =  'Dataset/transferlearning_forklift_trainlabels.pt'
forklift_finetunedataset_pth =  'Dataset/transferlearning_forklift_finetunedataset.pt'
forklift_finetunelabels_pth =  'Dataset/transferlearning_forklift_finetunelabels.pt'
# 无人机数据集路径
uav_traindataset_pth = 'Dataset/transferlearning_uav_traindataset.pt'
uav_trainlabels_pth = 'Dataset/transferlearning_uav_trainlabels.pt'
uav_finetunedataset_pth = 'Dataset/transferlearning_uav_finetunedataset.pt'
uav_finetunelabels_pth = 'Dataset/transferlearning_uav_finetunelabels.pt'

def download_data(NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE):
    # 生成叉车数据集
    forklift_train_x, forklift_train_y, _ = generate_forklift_data(num_samples=NUM_SAMPLES_TRAIN, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)
    torch.save(forklift_train_x, forklift_traindataset_pth)
    torch.save(forklift_train_y, forklift_trainlabels_pth)
    for i in range(test_num):
        forklift_test_x, forklift_test_y, forklift_devicefinger_list = generate_forklift_data(num_samples=NUM_SAMPLES_TEST, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)
        forklift_testdataset_pth = f'Dataset/transferlearning_forklift_testdataset{i}.pt'
        forklift_testlabels_pth =  f'Dataset/transferlearning_forklift_testlabels{i}.pt'
        forklift_testdevicefinger_pth = f'Dataset/forklift_devicefinger_list{i}.json'
        torch.save(forklift_test_x, forklift_testdataset_pth)
        torch.save(forklift_test_y, forklift_testlabels_pth)
        with open(forklift_testdevicefinger_pth, 'w') as json_file:
            json.dump(forklift_devicefinger_list, json_file)
    forklift_finetune_x, forklift_finetune_y, _ = generate_forklift_data(num_samples=NUM_SAMPLES_FINETUNE, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)
    torch.save(forklift_finetune_x, forklift_finetunedataset_pth)
    torch.save(forklift_finetune_y, forklift_finetunelabels_pth)
    # 生成无人机数据集
    uav_train_x, uav_train_y, _ = generate_uav_data(num_samples=NUM_SAMPLES_TRAIN, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)
    torch.save(uav_train_x, uav_traindataset_pth)
    torch.save(uav_train_y, uav_trainlabels_pth)
    for i in range(test_num):
        uav_test_x, uav_test_y, uav_devicefinger_list = generate_uav_data(num_samples=NUM_SAMPLES_TEST, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)
        uav_testdataset_pth = f'Dataset/transferlearning_uav_testdataset{i}.pt'
        uav_testlabels_pth = f'Dataset/transferlearning_uav_testlabels{i}.pt'
        uav_testdevicefinger_pth = f'Dataset/uav_devicefinger_list{i}.json'
        torch.save(uav_test_x, uav_testdataset_pth)
        torch.save(uav_test_y, uav_testlabels_pth)
        with open(uav_testdevicefinger_pth, 'w') as json_file:
            json.dump(uav_devicefinger_list, json_file)
    uav_finetune_x,  uav_finetune_y, _ = generate_uav_data(num_samples=NUM_SAMPLES_FINETUNE, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)                                              
    torch.save(uav_finetune_x, uav_finetunedataset_pth)
    torch.save(uav_finetune_y, uav_finetunelabels_pth)

def main():
     # 初始化 LSTM 模型
    car_lstm = run.load_model()
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"云端：使用设备 {device}")
    car_lstm = car_lstm.to(device)
    lstm = lstm.to(device)

    # 加载叉车数据集
    forklift_train_x = torch.load(forklift_traindataset_pth)
    forklift_train_y = torch.load(forklift_trainlabels_pth)
    forklift_train_x, forklift_train_y = forklift_train_x.to(device), forklift_train_y.to(device)
    forklift_test_x_list = []
    forklift_test_y_list = []
    for i in range(test_num):
        forklift_testdataset_pth = f'Dataset/transferlearning_forklift_testdataset{i}.pt'
        forklift_testlabels_pth =  f'Dataset/transferlearning_forklift_testlabels{i}.pt'
        forklift_test_x = torch.load(forklift_testdataset_pth)
        forklift_test_y =  torch.load(forklift_testlabels_pth)
        forklift_test_x, forklift_test_y = forklift_test_x.to(device), forklift_test_y.to(device)
        forklift_test_x_list.append(forklift_test_x)
        forklift_test_y_list.append(forklift_test_y)
    forklift_finetune_x = torch.load(forklift_finetunedataset_pth)
    forklift_finetune_y = torch.load(forklift_finetunelabels_pth)
    forklift_finetune_x, forklift_finetune_y = forklift_finetune_x.to(device), forklift_finetune_y.to(device)

    # 加载无人机数据集
    uav_train_x = torch.load(uav_traindataset_pth)
    uav_train_y = torch.load(uav_trainlabels_pth)
    uav_train_x, uav_train_y = uav_train_x.to(device), uav_train_y.to(device)
    uav_test_x_list = []
    uav_test_y_list = []
    for i in range(test_num):
        uav_testdataset_pth = f'Dataset/transferlearning_uav_testdataset{i}.pt'
        uav_testlabels_pth = f'Dataset/transferlearning_uav_testlabels{i}.pt'
        uav_test_x = torch.load(uav_testdataset_pth)
        uav_test_y =  torch.load(uav_testlabels_pth)
        uav_test_x, uav_test_y = uav_test_x.to(device), uav_test_y.to(device)
        uav_test_x_list.append(uav_test_x)
        uav_test_y_list.append(uav_test_y)
    uav_finetune_x = torch.load(uav_finetunedataset_pth)
    uav_finetune_y = torch.load(uav_finetunelabels_pth)
    uav_finetune_x, uav_finetune_y = uav_finetune_x.to(device), uav_finetune_y.to(device) 

    # 设备指纹
    # 从 JSON 文件中读取
    total_device_test_list = []
    for i in range(test_num):
        forklift_testdevicefinger_pth = f'Dataset/forklift_devicefinger_list{i}.json'
        with open(forklift_testdevicefinger_pth, 'r') as json_file:
            forklift_devicefinger_list = json.load(json_file)
        total_device_test_list += forklift_devicefinger_list
    for i in range(test_num):
        uav_testdevicefinger_pth = f'Dataset/uav_devicefinger_list{i}.json'
        with open(uav_testdevicefinger_pth, 'r') as json_file:
            uav_devicefinger_list = json.load(json_file)
        total_device_test_list += uav_devicefinger_list
    ft_acc = sum(1 for k in total_device_test_list if k == "none") / len(total_device_test_list) + (1 - PER_POSITIVE)

 
    # 智能叉车模型重新训练
    start_time = time.time()
    forklift_lstm, _, _, _, _ = run.train_model(
        lstm, forklift_train_x, forklift_train_y, TRAIN_EPOCHS)
    end_time = time.time()
    train_time_forklift = end_time-start_time
    test_loss_forklift1 = 0
    test_acc_forklift1 = 0
    for forklift_test_x,forklift_test_y in zip(forklift_test_x_list, forklift_test_y_list):
        test_loss_forklift, test_acc_forklift, _, _, _ = run.test_model(forklift_lstm, forklift_test_x, forklift_test_y)
        test_loss_forklift1 += test_loss_forklift
        test_acc_forklift1 += test_acc_forklift
    test_loss_forklift1 /= test_num
    test_acc_forklift1 /= test_num
    # 小车模型直接用于叉车
    test_loss_forklift2 = 0
    test_acc_forklift2 = 0
    for forklift_test_x,forklift_test_y in zip(forklift_test_x_list, forklift_test_y_list):
        test_loss_forklift, test_acc_forklift, _, _, _ = run.test_model(car_lstm, forklift_test_x, forklift_test_y)
        test_loss_forklift2 += test_loss_forklift 
        test_acc_forklift2 += test_acc_forklift
    test_acc_forklift2 /= test_num
    test_loss_forklift2 /= test_num
    # 智能叉车微调
    start_time = time.time()
    forklift_finetune_lstm, _, _, _, _ = run.train_model(
        car_lstm, forklift_finetune_x, forklift_finetune_y, SFT_EPOCHS)
    end_time = time.time()
    finetune_time_forklift = end_time-start_time
    test_loss_forklift3 = 0
    test_acc_forklift3 = 0 
    for forklift_test_x, forklift_test_y in zip(forklift_test_x_list, forklift_test_y_list):
        test_loss_forklift, test_acc_forklift, _, _, _ = run.test_model(forklift_finetune_lstm, forklift_test_x, forklift_test_y)
        test_loss_forklift3 += test_loss_forklift 
        test_acc_forklift3 += test_acc_forklift
    test_loss_forklift3 /= test_num
    test_acc_forklift3 /= test_num 


    # 无人机重新训练
    start_time = time.time()
    uav_lstm, _, _, _, _ = run.train_model(
        lstm, uav_train_x, uav_train_y, TRAIN_EPOCHS)
    end_time = time.time()
    train_time_uav = end_time-start_time
    test_loss_uav1 = 0
    test_acc_uav1 = 0
    for uav_test_x, uav_test_y in zip(uav_test_x_list, uav_test_y_list):
        test_loss_uav, test_acc_uav, _, _, _ = run.test_model(uav_lstm, uav_test_x, uav_test_y)
        test_loss_uav1 += test_loss_uav
        test_acc_uav1 += test_acc_uav
    test_loss_uav1 /= test_num
    test_acc_uav1 /= test_num
    # 小车模型直接用于无人机
    test_loss_uav2 = 0
    test_acc_uav2 = 0
    for uav_test_x, uav_test_y in zip(uav_test_x_list, uav_test_y_list):
        test_loss_uav, test_acc_uav, _, _, _ = run.test_model(car_lstm, uav_test_x, uav_test_y)
        test_loss_uav2 += test_loss_uav
        test_acc_uav2 += test_acc_uav
    test_loss_uav2 /= test_num
    test_acc_uav2 /= test_num
    # 无人机微调
    start_time = time.time()
    uav_finetune_lstm, _, _, _, _ = run.train_model(
        car_lstm, uav_finetune_x, uav_finetune_y, SFT_EPOCHS)
    end_time = time.time()
    finetune_time_uav = end_time-start_time
    test_loss_uav3 = 0
    test_acc_uav3 = 0
    for uav_test_x, uav_test_y in zip(uav_test_x_list, uav_test_y_list):    
        test_loss_uav, test_acc_uav, _, _, _ = run.test_model(uav_finetune_lstm, uav_test_x, uav_test_y)
        test_loss_uav3 += test_loss_uav
        test_acc_uav3 += test_acc_uav
    test_loss_uav3 /= test_num
    test_acc_uav3 /= test_num
    
    # 打印
    print(f"设备指纹, 准确率: {ft_acc}")
    print(f"叉车模型重新训练完成, 准确率: {test_acc_forklift1}, 损失值：{test_loss_forklift1}, 训练时间：{train_time_forklift}")
    print(f"小车模型直接用于叉车, 准确率: {test_acc_forklift2}, 损失值：{test_loss_forklift2}")
    print(f"叉车模型微调完成, 准确率: {test_acc_forklift3}, 损失值：{test_loss_forklift3}, 训练时间：{finetune_time_forklift}")
    print(f"无人机模型重新训练完成, 准确率: {test_acc_uav1}, 损失值：{test_loss_uav1}, 训练时间：{train_time_uav}")
    print(f"小车模型直接用于无人机, 准确率: {test_acc_uav2}, 损失值：{test_loss_uav2}")
    print(f"无人机模型微调完成, 准确率: {test_acc_uav3}, 损失值：{test_loss_uav3}, 训练时间：{finetune_time_uav}")

if __name__ == "__main__":
    download_data(NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE)
    main()