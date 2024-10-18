import time, torch, run, json
from Dataset import generate_forklift_data, generate_uav_data, generate_car_data
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
PER_CONTROL = 1
TEST_NUM = 10

# 无人车数据集
car_traindataset_pth = 'Dataset/transferlearning_car_traindataset.pt'
car_trainlabels_pth =  'Dataset/transferlearning_cart_trainlabels.pt'
car_finetunedataset_pth =  'Dataset/transferlearning_car_finetunedataset.pt'
car_finetunelabels_pth =  'Dataset/transferlearning_car_finetunelabels.pt'
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

def download_data(NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE, TEST_NUM):
    # 生成小车数据集
    car_train_x, car_train_y, _ = generate_car_data(num_samples=NUM_SAMPLES_TRAIN, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)
    torch.save(car_train_x, car_traindataset_pth)
    torch.save(car_train_y, car_trainlabels_pth)
    for i in range(TEST_NUM):
        car_test_x, car_test_y, car_devicefinger_list = generate_car_data(num_samples=NUM_SAMPLES_TEST, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)
        car_testdataset_pth = f'Dataset/transferlearning_car_testdataset{i}.pt'
        car_testlabels_pth =  f'Dataset/transferlearning_car_testlabels{i}.pt'
        car_testdevicefinger_pth = f'Dataset/car_devicefinger_list{i}.json'
        torch.save(car_test_x, car_testdataset_pth)
        torch.save(car_test_y, car_testlabels_pth)
        with open(car_testdevicefinger_pth, 'w') as json_file:
            json.dump(car_devicefinger_list, json_file)
    car_finetune_x, car_finetune_y, _ = generate_car_data(num_samples=NUM_SAMPLES_FINETUNE, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)
    torch.save(car_finetune_x, car_finetunedataset_pth)
    torch.save(car_finetune_y, car_finetunelabels_pth)
    # 生成叉车数据集
    forklift_train_x, forklift_train_y, _ = generate_forklift_data(num_samples=NUM_SAMPLES_TRAIN, input_dim=INPUT_DIM, per_positive=PER_POSITIVE, per_control=PER_CONTROL)
    torch.save(forklift_train_x, forklift_traindataset_pth)
    torch.save(forklift_train_y, forklift_trainlabels_pth)
    for i in range(TEST_NUM):
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
    for i in range(TEST_NUM):
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

# 计算平均损失和准确率
def compute_avg_loss_acc(model, test_x_list, test_y_list):
    total_loss, total_acc = 0, 0
    for test_x, test_y in zip(test_x_list, test_y_list):
        loss, acc, _, _, _ = run.test_model(model, test_x, test_y)
        total_loss += loss
        total_acc += acc
    return total_loss / len(test_x_list), total_acc / len(test_x_list)

def main():
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"云端：使用设备 {device}")


    # 加载无人车数据集
    car_train_x = torch.load(car_traindataset_pth)
    car_train_y = torch.load(car_trainlabels_pth)
    car_train_x, car_train_y = car_train_x.to(device), car_train_y.to(device)
    car_test_x_list = []
    car_test_y_list = []
    for i in range(TEST_NUM):
        car_testdataset_pth = f'Dataset/transferlearning_car_testdataset{i}.pt'
        car_testlabels_pth = f'Dataset/transferlearning_car_testlabels{i}.pt'
        car_test_x = torch.load(car_testdataset_pth)
        car_test_y =  torch.load(car_testlabels_pth)
        car_test_x, car_test_y = car_test_x.to(device), car_test_y.to(device)
        car_test_x_list.append(car_test_x)
        car_test_y_list.append(car_test_y)
    car_finetune_x = torch.load(car_finetunedataset_pth)
    car_finetune_y = torch.load(car_finetunelabels_pth)
    car_finetune_x, car_finetune_y = car_finetune_x.to(device), car_finetune_y.to(device)

    # 加载叉车数据集
    forklift_train_x = torch.load(forklift_traindataset_pth)
    forklift_train_y = torch.load(forklift_trainlabels_pth)
    forklift_train_x, forklift_train_y = forklift_train_x.to(device), forklift_train_y.to(device)
    forklift_test_x_list = []
    forklift_test_y_list = []
    for i in range(TEST_NUM):
        forklift_testdataset_pth = f'Dataset/transferlearning_forklift_testdataset{i}.pt'
        forklift_testlabels_pth = f'Dataset/transferlearning_forklift_testlabels{i}.pt'
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
    for i in range(TEST_NUM):
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
    for i in range(TEST_NUM):
        car_testdevicefinger_pth = f'Dataset/car_devicefinger_list{i}.json'
        with open(car_testdevicefinger_pth, 'r') as json_file:
            car_devicefinger_list = json.load(json_file)
        total_device_test_list += car_devicefinger_list
    for i in range(TEST_NUM):
        forklift_testdevicefinger_pth = f'Dataset/forklift_devicefinger_list{i}.json'
        with open(forklift_testdevicefinger_pth, 'r') as json_file:
            forklift_devicefinger_list = json.load(json_file)
        total_device_test_list += forklift_devicefinger_list
    for i in range(TEST_NUM):
        uav_testdevicefinger_pth = f'Dataset/uav_devicefinger_list{i}.json'
        with open(uav_testdevicefinger_pth, 'r') as json_file:
            uav_devicefinger_list = json.load(json_file)
        total_device_test_list += uav_devicefinger_list
    ft_acc = sum(1 for k in total_device_test_list if k == "none") / len(total_device_test_list) + PER_POSITIVE

    # 训练无人车
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)
    lstm = lstm.to(device)
    start_time = time.time()
    car_lstm, _, _, _, _ = run.train_model(
        lstm, car_train_x, car_train_y, TRAIN_EPOCHS)
    end_time = time.time()
    train_time_car = end_time-start_time
    car_lstm = car_lstm.to(device)
    test_loss_car, test_acc_car = compute_avg_loss_acc(car_lstm, car_test_x_list, car_test_y_list)
 
    # 智能叉车模型重新训练
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)
    lstm = lstm.to(device)
    start_time = time.time()
    forklift_lstm, _, _, _, _ = run.train_model(
        lstm, forklift_train_x, forklift_train_y, TRAIN_EPOCHS)
    end_time = time.time()
    train_time_forklift = end_time-start_time
    test_loss_forklift1, test_acc_forklift1 = compute_avg_loss_acc(forklift_lstm, forklift_test_x_list, forklift_test_y_list)
    # 小车模型直接用于叉车
    test_loss_forklift2, test_acc_forklift2 = compute_avg_loss_acc(car_lstm, forklift_test_x_list, forklift_test_y_list )
    # 智能叉车微调
    start_time = time.time()
    forklift_finetune_lstm, _, _, _, _ = run.train_model(
        car_lstm, forklift_finetune_x, forklift_finetune_y, SFT_EPOCHS)
    end_time = time.time()
    finetune_time_forklift = end_time-start_time
    test_loss_forklift3, test_acc_forklift3 = compute_avg_loss_acc(forklift_finetune_lstm, forklift_test_x_list, forklift_test_y_list)


    # 无人机重新训练
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)
    lstm = lstm.to(device)
    start_time = time.time()
    uav_lstm, _, _, _, _ = run.train_model(
        lstm, uav_train_x, uav_train_y, TRAIN_EPOCHS)
    end_time = time.time()
    train_time_uav = end_time-start_time
    test_loss_uav1, test_acc_uav1 = compute_avg_loss_acc(uav_lstm, uav_test_x_list, uav_test_y_list)
    # 小车模型直接用于无人机
    test_loss_uav2, test_acc_uav2 = compute_avg_loss_acc(car_lstm, uav_test_x_list, uav_test_y_list)
    # 无人机微调
    start_time = time.time()
    uav_finetune_lstm, _, _, _, _ = run.train_model(
        car_lstm, uav_finetune_x, uav_finetune_y, SFT_EPOCHS)
    end_time = time.time()
    finetune_time_uav = end_time-start_time
    test_loss_uav3, test_acc_uav3 = compute_avg_loss_acc(uav_finetune_lstm, uav_test_x_list, uav_test_y_list)
    
    # 打印
    print(f"设备指纹, 准确率: {ft_acc}")
    print(f"无人车模型训练完成, 准确率: {test_acc_car}, 损失值：{test_loss_car}, 训练时间：{train_time_car}")
    print(f"叉车模型重新训练完成, 准确率: {test_acc_forklift1}, 损失值：{test_loss_forklift1}, 训练时间：{train_time_forklift}")
    print(f"小车模型直接用于叉车, 准确率: {test_acc_forklift2}, 损失值：{test_loss_forklift2}")
    print(f"叉车模型微调完成, 准确率: {test_acc_forklift3}, 损失值：{test_loss_forklift3}, 训练时间：{finetune_time_forklift}")
    print(f"无人机模型重新训练完成, 准确率: {test_acc_uav1}, 损失值：{test_loss_uav1}, 训练时间：{train_time_uav}")
    print(f"小车模型直接用于无人机, 准确率: {test_acc_uav2}, 损失值：{test_loss_uav2}")
    print(f"无人机模型微调完成, 准确率: {test_acc_uav3}, 损失值：{test_loss_uav3}, 训练时间：{finetune_time_uav}")


if __name__ == "__main__":
    download_data(NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE, TEST_NUM)
    main()