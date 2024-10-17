import time, torch, run, json, os
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
test_num = 10  # 生成测试数据集数量，最后求平均

# 通用路径生成
def generate_file_path(prefix, dataset_type, index=None, ext='.pt'):
    if index is not None:
        return f'{prefix}/transferlearning_{dataset_type}{index}{ext}'
    return f'{prefix}/transferlearning_{dataset_type}{ext}'

# 生成并保存数据集
def generate_and_save_data(generate_func, prefix, num_samples_train, num_samples_test, num_samples_finetune, input_dim, per_positive):
    train_x, train_y, _ = generate_func(num_samples=num_samples_train, input_dim=input_dim, per_positive=per_positive, per_control=PER_CONTROL)
    torch.save(train_x, generate_file_path(prefix, 'traindataset'))
    torch.save(train_y, generate_file_path(prefix, 'trainlabels'))
    
    for i in range(test_num):
        test_x, test_y, device_finger_list = generate_func(num_samples=num_samples_test, input_dim=input_dim, per_positive=per_positive, per_control=PER_CONTROL)
        torch.save(test_x, generate_file_path(prefix, 'testdataset', i))
        torch.save(test_y, generate_file_path(prefix, 'testlabels', i))
        with open(generate_file_path(prefix, 'devicefinger_list', i, ext='.json'), 'w') as json_file:
            json.dump(device_finger_list, json_file)
    
    finetune_x, finetune_y, _ = generate_func(num_samples=num_samples_finetune, input_dim=input_dim, per_positive=per_positive, per_control=PER_CONTROL)
    torch.save(finetune_x, generate_file_path(prefix, 'finetunedataset'))
    torch.save(finetune_y, generate_file_path(prefix, 'finetunelabels'))

# 加载数据集
def load_datasets(prefix, device):
    train_x = torch.load(generate_file_path(prefix, 'traindataset')).to(device)
    train_y = torch.load(generate_file_path(prefix, 'trainlabels')).to(device)
    
    test_x_list = []
    test_y_list = []
    for i in range(test_num):
        test_x = torch.load(generate_file_path(prefix, 'testdataset', i)).to(device)
        test_y = torch.load(generate_file_path(prefix, 'testlabels', i)).to(device)
        test_x_list.append(test_x)
        test_y_list.append(test_y)
    
    finetune_x = torch.load(generate_file_path(prefix, 'finetunedataset')).to(device)
    finetune_y = torch.load(generate_file_path(prefix, 'finetunelabels')).to(device)
    
    return train_x, train_y, test_x_list, test_y_list, finetune_x, finetune_y

# 计算平均损失和准确率
def compute_avg_loss_acc(model, test_x_list, test_y_list):
    total_loss, total_acc = 0, 0
    for test_x, test_y in zip(test_x_list, test_y_list):
        loss, acc, _, _, _ = run.test_model(model, test_x, test_y)
        total_loss += loss
        total_acc += acc
    return total_loss / len(test_x_list), total_acc / len(test_x_list)

# 下载数据
def download_data():
    generate_and_save_data(generate_forklift_data, 'Dataset/forklift', NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE)
    generate_and_save_data(generate_uav_data, 'Dataset/uav', NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE)

# 训练、测试和微调
def train_test_finetune(model, train_x, train_y, test_x_list, test_y_list, finetune_x, finetune_y, device):
    start_time = time.time()
    trained_model, _, _, _, _ = run.train_model(model, train_x, train_y, TRAIN_EPOCHS)
    train_time = time.time() - start_time
    
    avg_loss, avg_acc = compute_avg_loss_acc(trained_model, test_x_list, test_y_list)
    
    start_time = time.time()
    finetuned_model, _, _, _, _ = run.train_model(trained_model, finetune_x, finetune_y, SFT_EPOCHS)
    finetune_time = time.time() - start_time
    
    finetune_loss, finetune_acc = compute_avg_loss_acc(finetuned_model, test_x_list, test_y_list)
    
    return avg_loss, avg_acc, train_time, finetune_loss, finetune_acc, finetune_time

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    car_lstm = run.load_model().to(device)
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE).to(device)
    
    # 加载叉车数据
    forklift_train_x, forklift_train_y, forklift_test_x_list, forklift_test_y_list, forklift_finetune_x, forklift_finetune_y = load_datasets('Dataset/forklift', device)
    # 叉车模型训练和测试
    fork_loss, fork_acc, train_time_fork, finetune_loss_fork, finetune_acc_fork, finetune_time_fork = train_test_finetune(
        lstm, forklift_train_x, forklift_train_y, forklift_test_x_list, forklift_test_y_list, forklift_finetune_x, forklift_finetune_y, device
    )
    
    # 加载无人机数据
    uav_train_x, uav_train_y, uav_test_x_list, uav_test_y_list, uav_finetune_x, uav_finetune_y = load_datasets('Dataset/uav', device)
    # 无人机模型训练和测试
    uav_loss, uav_acc, train_time_uav, finetune_loss_uav, finetune_acc_uav, finetune_time_uav = train_test_finetune(
        lstm, uav_train_x, uav_train_y, uav_test_x_list, uav_test_y_list, uav_finetune_x, uav_finetune_y, device
    )
    
    # 小车模型直接用于叉车和无人机
    fork_loss_car, fork_acc_car, _, _, _, _ = train_test_finetune(car_lstm, forklift_train_x, forklift_train_y, forklift_test_x_list, forklift_test_y_list, forklift_finetune_x, forklift_finetune_y, device)
    uav_loss_car, uav_acc_car, _, _, _, _ = train_test_finetune(car_lstm, uav_train_x, uav_train_y, uav_test_x_list, uav_test_y_list, uav_finetune_x, uav_finetune_y, device)
    
    # 打印结果
    print(f"叉车模型重新训练完成, 准确率: {fork_acc}, 损失值：{fork_loss}, 训练时间：{train_time_fork}")
    print(f"小车模型直接用于叉车, 准确率: {fork_acc_car}, 损失值：{fork_loss_car}")
    print(f"叉车模型微调完成, 准确率: {finetune_acc_fork}, 损失值：{finetune_loss_fork}, 微调时间：{finetune_time_fork}")
    print(f"无人机模型重新训练完成, 准确率: {uav_acc}, 损失值：{uav_loss}, 训练时间：{train_time_uav}")
    print(f"小车模型直接用于无人机, 准确率: {uav_acc_car}, 损失值：{uav_loss_car}")
    print(f"无人机模型微调完成, 准确率: {finetune_acc_uav}, 损失值：{finetune_loss_uav}, 微调时间：{finetune_time_uav}")

if __name__ == "__main__":
    # 下载数据
    # download_data()
    main()
