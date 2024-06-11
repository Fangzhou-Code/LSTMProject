import run
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import torch
import Dataset
import pickle
import random
import device
import route
import numpy as np

def generate_car_data(num_samples, input_dim, num_negative, route_number):
    # 初始化
    dataset = np.zeros((num_samples, input_dim, 3))
    labels = np.zeros((num_samples, 5), dtype=int)
    waiting_time = 6  # 待料状态下需要等待6秒
    device_list = []
    per_control = 0.1  # 控制数量占全部负样本的比例
    print("开始生成样本")
    
    # 路线起始点和终点
    start_point, end_point = route.get_route_coordinates(route_number)
    end_x = end_point[0]
    end_y = end_point[1]
    start_x = start_point[0]
    start_y = start_point[1]
    
    for i in range(num_samples):
        # 随机初始化小车的静态属性和位置和状态
        all_data = device.generate_vehicle_data()
        fingerpoint = device.get_attribute(all_data, 'fingerprint', 'fingerprint')
        device_list.append(fingerpoint)
        
        distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5 
        speed_x = (end_x - start_x) / distance 
        speed_y = (end_y - start_y) / distance
        pos_x = random.uniform(start_x, end_x)
        pos_y = start_y + speed_y * ((end_x - start_x) / speed_x)
        
        if i < num_negative:
             # 生成负样本
            print(f"生成负样本 {i + 1}/{num_negative}/{num_samples}")
            control_or_impersonation = random.choice(["control"]) if random.random() <= per_control else random.choice(["Impersonation"])
            if control_or_impersonation == "Impersonation":
                device_list[-1] = "none"  # 由于没用数据库，判定当设备指纹=none的时候设备遭到冒充
            status = random.choice(["seal", "maintenance", "depleted"])
            if status == "seal":
                power = random.uniform(1, 100)
                labels[i, 2] = 1  # seal=00100
                for j in range(input_dim):
                    dataset[i, j, 0] = pos_x
                    dataset[i, j, 1] = pos_y
                    dataset[i, j, 2] = power
            elif status == "maintenance":
                power = random.uniform(50, 100)
                labels[i, 3] = 1  # maintenance=00010
                stop_time = random.randint(1, input_dim - 1)
                for j in range(input_dim):
                    if j < stop_time:
                        dataset[i, j, 0] = pos_x - (stop_time - j) * speed_x
                        dataset[i, j, 1] = pos_y - (stop_time - j) * speed_y
                        displacement = ((dataset[i, stop_time, 0] - dataset[i, j, 0]) ** 2 + 
                                        (dataset[i, stop_time, 1] - dataset[i, j, 1]) ** 2) ** 0.5
                        power -= displacement * random.uniform(0.05, 0.1)
                        dataset[i, j, 2] = power
                    else:
                        dataset[i, j, 0] = pos_x
                        dataset[i, j, 1] = pos_y
                        dataset[i, j, 2] = power
            elif status == "depleted":
                labels[i, 4] = 1  # 没电状态label=00001
                consume_power = []  # 消耗电量
                stop_time = random.randint(1, input_dim - 1)
                for j in range(input_dim):
                    if j <= stop_time:
                        if j == 0:
                            dataset[i, j, 0] = pos_x
                            dataset[i, j, 1] = pos_y
                        else:
                            dataset[i, j, 0] = dataset[i, j - 1, 0] + speed_x
                            dataset[i, j, 1] = dataset[i, j - 1, 1] + speed_y
                            displacement = ((dataset[i, j, 0] - dataset[i, j - 1, 0]) ** 2 + 
                                            (dataset[i, j, 1] - dataset[i, j - 1, 1]) ** 2) ** 0.5
                            consume_power.append(displacement * random.uniform(0.05, 0.1))
                    else:
                        dataset[i, j, 0] = dataset[i, stop_time, 0]
                        dataset[i, j, 1] = dataset[i, stop_time, 1]
                        dataset[i, j, 2] = 0
                power = sum(consume_power)
                dataset[i, 0, 2] = power
                for j in range(stop_time):
                    power -= consume_power[j]
                    dataset[i, j + 1, 2] = max(0, power)
            
        else:
            # 生成正样本
            print(f"生成正样本 {i + 1}/{num_samples-num_negative}/{num_samples}")
            status = random.choice(["running", "waiting"])
            if status == "running":
                power = random.uniform(50, 100)
                labels[i, 0] = 1  # running=10000 
                for j in range(input_dim):
                    if j == 0:
                        dataset[i, j, 0] = pos_x
                        dataset[i, j, 1] = pos_y
                        dataset[i, j, 2] = power
                    else:
                        dataset[i, j, 0] = dataset[i, j - 1, 0] + speed_x
                        dataset[i, j, 1] = dataset[i, j - 1, 1] + speed_y
                        displacement = ((dataset[i, j, 0] - dataset[i, j - 1, 0]) ** 2 + 
                                        (dataset[i, j, 1] - dataset[i, j - 1, 1]) ** 2) ** 0.5
                        power -= displacement * random.uniform(0.05, 0.1)
                        dataset[i, j, 2] = power
            else:
                power = random.uniform(50, 100)
                labels[i, 1] = 1  # waiting = 01000
                stop_time = random.randint(0, input_dim - 1)
                for j in range(input_dim):
                    if j < stop_time:
                        dataset[i, j, 0] = end_x - (stop_time - j) * speed_x
                        dataset[i, j, 1] = end_y - (stop_time - j) * speed_y
                        displacement = ((dataset[i, stop_time, 0] - dataset[i, j, 0]) ** 2 + 
                                        (dataset[i, stop_time, 1] - dataset[i, j, 1]) ** 2) ** 0.5
                        power -= displacement * random.uniform(0.05, 0.1)
                        dataset[i, j, 2] = power 
                    else:
                        if j > stop_time + waiting_time:
                            dataset[i, j, 0] = dataset[i, j - 1, 0] + speed_x
                            dataset[i, j, 1] = dataset[i, j - 1, 1] + speed_y
                            displacement = ((dataset[i, stop_time, 0] - dataset[i, j, 0]) ** 2 + 
                                            (dataset[i, stop_time, 1] - dataset[i, j, 1]) ** 2) ** 0.5
                            power -= displacement * random.uniform(0.05, 0.1)
                            dataset[i, j, 2] = power
                        else:
                            dataset[i, j, 0] = end_x
                            dataset[i, j, 1] = end_y
                            dataset[i, j, 2] = power
    
    dataset_ = torch.tensor(dataset, dtype=torch.float32)
    labels_ = torch.tensor(labels, dtype=torch.float32)
    print("...generate_car_data run Finished...")
    return dataset_, labels_, device_list




def main():
    # 初始化
    input_dim = 20
    lstm, train_x, train_y, device_train_list = run.initialize_model_data(3, input_dim, 0)
    train_epochs = 500

    # 检查是否有 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm = lstm.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # 训练
    run.save_model(lstm)
    print("...Training Finished...")    

    # 测试
    lstm = run.load_model().to(device)
    device_id = "device123" 
    manufacturer = "Example Inc."
    device_auth = run.DeviceAuthentication(device_id, manufacturer) # 实例化设备验证类

    # 定义不同类型负样本的比例
    negative_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    ft_false_positive_rates = [0.6, 0.55, 0.53, 0.51, 0.5]  # 设备指纹误判率

    # 测试结果记录列表
    results = []
    num_workshops = 3
    num_avg = 1
    
    for ratio in negative_ratios:
        total_test_acc = 0
        total_test_loss = 0
        total_device_test_list = []
        false_positives = 0
        num_negative_samples = 0
        
        for _ in range(num_avg):  # 进行10次测试
            num_samples = 100  # 总样本数
            num_negative_samples_current = round(num_samples * ratio / 3) # 根据比例计算负样本数量
            for i in range(num_workshops):
                # 生成测试数据
                test_x, test_y, device_test_list = generate_car_data(num_samples, input_dim, num_negative_samples_current, i+1)

                test_x = test_x.to(device)
                test_y = test_y.to(device)

                # 测试模型
                test_loss, test_acc, pred_labels = run.test_model(lstm, test_x, test_y)
                print("...Test Finished...") 
                print("test_acc=", test_acc.item())
                print("test_loss=", test_loss.item())
                print("Number of test samples:", num_samples)  

                # 累加测试精度和测试损失
                total_test_acc += test_acc.item()
                total_test_loss += test_loss.item()
                total_device_test_list += device_test_list
                
                # 计算误判
                negative_samples_indices = np.where(test_y.cpu().numpy()[:, 2:].sum(axis=1) > 0)[0]
                false_positives += (pred_labels[negative_samples_indices, 0].cpu().numpy() == 1).sum() + (pred_labels[negative_samples_indices, 1].cpu().numpy() == 1).sum()
                num_negative_samples += len(negative_samples_indices)

                # 设备认证
                device_auth.authenticate_device(pred_labels)

                # 更新样本数
                num_samples = round(num_samples * test_acc.item())

        # 记录测试结果
        average_test_acc = total_test_acc / (num_workshops * num_avg)
        average_test_loss = total_test_loss / (num_workshops * num_avg)
        ft_acc = sum(1 for k in total_device_test_list if k == "none") / len(total_device_test_list) + (1 - ratio)
        false_positive_rate = false_positives / num_negative_samples
        results.append((ratio, ft_acc, average_test_acc, false_positive_rate))

    # 输出
    for ratio, ft_acc, average_test_acc, false_positive_rate in results:
        print(f"Negative Sample Ratio: {ratio}")
        print(f"设备指纹准确率: {ft_acc}")
        print(f"LSTM准确率: {average_test_acc}")
        print(f"误判率: {false_positive_rate}")
        print()

    # 画设备指纹和LSTM准确率柱状图
    fig, ax = plt.subplots()

    bar_width = 0.35  # 柱子的宽度
    index = np.arange(len(negative_ratios))  # 每个比例的索引

    ft_accs = [result[1] for result in results]
    lstm_accs = [result[2] for result in results]

    ax.bar(index - bar_width, ft_accs, bar_width, label='设备指纹', color='#94CE87')  # 向左移动bar_width
    ax.bar(index, lstm_accs, bar_width, label='LSTM', color='#EE634B')  # 原始位置

    ax.set_xlabel('Negative Sample Ratio')
    ax.set_ylabel('Accuracy')
    ax.set_title('身份验证准确率')
    ax.set_xticks(index - bar_width / 2)  # 调整xticks的位置
    ax.set_xticklabels([f'{ratio:.1%}' for ratio in negative_ratios])  # 格式化横坐标标签为百分比
    ax.legend(['设备指纹', 'LSTM'], loc='upper right')

    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.savefig('./results/accuracy_comparison_bar.png')
    plt.show()

    # 画误判率柱状图
    fig, ax = plt.subplots()

    false_positive_rates = [(1-result[3]) for result in results]

    ax.bar(index - bar_width, ft_false_positive_rates, bar_width, label='设备指纹', color='#94CE87')  # 原始位置
    ax.bar(index , false_positive_rates, bar_width, label='LSTM', color='#EE634B')  # 向左移动bar_width

    ax.set_xlabel('Negative Sample Ratio')
    ax.set_ylabel('False Positive Rate')
    ax.set_title('伪造设备检出率')
    ax.set_xticks(index - bar_width / 2)  # 调整xticks的位置
    ax.set_xticklabels([f'{ratio:.1%}' for ratio in negative_ratios])  # 格式化横坐标标签为百分比
    ax.legend(['设备指纹', 'LSTM'], loc='upper right')

    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.savefig('./results/false_positive_rate_bar.png')
    plt.show()

if __name__ == "__main__":
    main()
