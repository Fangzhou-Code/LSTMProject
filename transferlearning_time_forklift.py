import time, torch, run, json
from Dataset import generate_forklift_data, generate_uav_data, generate_car_data
from Model import LSTM
import concurrent.futures

# 定义全局参数
INPUT_DIM = 10
TRAIN_EPOCHS = 1500
SFT_EPOCHS = 300
NUM_SAMPLES_TRAIN = 1000
NUM_SAMPLES_FINETUNE = 100
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 3
PRED_OUTPUT_SIZE = INPUT_SIZE
CLAS_OUTPUT_SIZE = 5
PER_POSITIVE = 0.2
PER_CONTROL = 1

# 数据生成部分使用线程池来并行生成叉车数据并记录时间
def download_forklift_data_in_parallel(NUM_SAMPLES_TRAIN, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE, num_forklift):
    total_train_time = 0
    total_finetune_time = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_forklift) as executor:
        # 记录训练集生成时间
        start_train_time = time.time()
        future_train_data = [executor.submit(generate_forklift_data, num_samples=NUM_SAMPLES_TRAIN, input_dim=INPUT_DIM, per_positive=PER_POSITIVE) for _ in range(num_forklift)]
        train_data = [future.result() for future in future_train_data]
        end_train_time = time.time()
        total_train_time = end_train_time - start_train_time
        
        # 记录微调集生成时间
        start_finetune_time = time.time()
        future_finetune_data = [executor.submit(generate_forklift_data, num_samples=NUM_SAMPLES_FINETUNE, input_dim=INPUT_DIM, per_positive=PER_POSITIVE) for _ in range(num_forklift)]
        finetune_data = [future.result() for future in future_finetune_data]
        end_finetune_time = time.time()
        total_finetune_time = end_finetune_time - start_finetune_time

    # 合并叉车训练和微调数据集
    forklift_train_x, forklift_train_y = merge_forklift_data(train_data)
    forklift_finetune_x, forklift_finetune_y = merge_forklift_data(finetune_data)

    return forklift_train_x, forklift_train_y, forklift_finetune_x, forklift_finetune_y, total_train_time, total_finetune_time

# 合并叉车数据集
def merge_forklift_data(data_list):
    # 假设 data_list 是多个 (train_x, train_y, _) 元组的列表，忽略第三个值
    train_x_list, train_y_list, _ = zip(*data_list)
    
    # 使用 torch.cat 将多个数据集合并成一个
    merged_train_x = torch.cat(train_x_list, dim=0)
    merged_train_y = torch.cat(train_y_list, dim=0)
    
    return merged_train_x, merged_train_y

def run_experiment(num_forklift):
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"云端：使用设备 {device}, num_forklift={num_forklift}")
    
    # 加载无人车数据集
    car_traindataset_pth = 'Dataset/transferlearning_car_traindataset.pt'
    car_trainlabels_pth = 'Dataset/transferlearning_cart_trainlabels.pt'
    car_train_x = torch.load(car_traindataset_pth)
    car_train_y = torch.load(car_trainlabels_pth)
    car_train_x, car_train_y = car_train_x.to(device), car_train_y.to(device)
    
    # 加载叉车数据集
    forklift_train_x, forklift_train_y, forklift_finetune_x, forklift_finetune_y, traindata_time_forklift, finetunedata_time_forklift = download_forklift_data_in_parallel(
        NUM_SAMPLES_TRAIN, NUM_SAMPLES_FINETUNE, INPUT_DIM, PER_POSITIVE, num_forklift)
    forklift_train_x, forklift_train_y = forklift_train_x.to(device), forklift_train_y.to(device)
    forklift_finetune_x, forklift_finetune_y = forklift_finetune_x.to(device), forklift_finetune_y.to(device)

    # 训练无人车
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)
    lstm = lstm.to(device)
    car_lstm, _, _, _, _ = run.train_model(lstm, car_train_x, car_train_y, TRAIN_EPOCHS)
    car_lstm = car_lstm.to(device)

    # 智能叉车模型重新训练
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)
    lstm = lstm.to(device)
    start_time = time.time()
    _, _, _, _, _ = run.train_model(lstm, forklift_train_x, forklift_train_y, TRAIN_EPOCHS)
    end_time = time.time()
    train_time_forklift = end_time - start_time
    
    # 智能叉车微调
    start_time = time.time()
    _, _, _, _, _ = run.train_model(car_lstm, forklift_finetune_x, forklift_finetune_y, SFT_EPOCHS)
    end_time = time.time()
    finetune_time_forklift = end_time - start_time

    # 返回训练时间和微调时间
    total_train_time = train_time_forklift + traindata_time_forklift
    total_finetune_time = finetune_time_forklift + finetunedata_time_forklift
    
    return train_time_forklift, finetune_time_forklift, total_train_time, total_finetune_time

def main():
    # 要测试的 num_forklift 数值列表
    num_forklift_values = [1, 5, 10, 15, 20, 25, 30, 40, 50]
    
    # 存储每个 num_forklift 的时间结果
    results = {}

    # 逐轮运行不同的 num_forklift
    for num_forklift in num_forklift_values:
        train_time_forklift, finetune_time_forklift, total_train_time, total_finetune_time = run_experiment(num_forklift)
        results[num_forklift] = ( train_time_forklift, finetune_time_forklift, total_train_time, total_finetune_time)
        print(f"num_forklift={num_forklift} 训练时间：{train_time_forklift} 秒, 微调时间{finetune_time_forklift} 秒, 总训练时间：{total_train_time} 秒, 总微调时间：{total_finetune_time} 秒")
    
    # 统一输出所有结果
    print("\n====== 总结结果 ======")
    for num_forklift, ( train_time_forklift, finetune_time_forklift, total_train_time, total_finetune_time) in results.items():
        print(f"num_forklift={num_forklift} 训练时间：{train_time_forklift} 秒, 微调时间：{finetune_time_forklift} 秒, 总训练时间：{total_train_time} 秒, 总微调时间：{total_finetune_time} 秒")
if __name__ == "__main__":
    main()
