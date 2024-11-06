import numpy as np
import random, device, route, torch
from Model import LSTM
import torch.optim as optim
import torch.nn.functional as F

def generate_car_data(samples_list): # samples_list 输入0和1组成的数组
    # 初始化
    input_dim = 10
    per_control = 0.5
    num_samples = len(samples_list)
    dataset = np.zeros((num_samples, input_dim, 4))
    labels = np.zeros((num_samples,5),dtype=int)
    waiting_time = 6 # 待料状态下需要等待6秒
    device_list = []
    print("开始生成样本")

    for i in range(num_samples):
        print(f"生成样本 {i + 1}/{num_samples}")
        # 随机路线
        route_number = random.randint(1,3)
        start_point, end_point = route.get_route_coordinates(route_number)
        start_x, start_y, start_z = start_point
        end_x, end_y, end_z = end_point

        # 随机初始化小车的静态属性和位置和状态
        all_data = device.generate_vehicle_data()
        fingerpoint = device.get_attribute(all_data, 'fingerprint', 'fingerprint')
        device_list.append(fingerpoint)
        distance = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5 
        speed_x = (end_x - start_x) / distance 
        speed_y = (end_y - start_y) / distance
        pos_x = random.uniform(start_x, end_x)
        pos_y = start_y + speed_y * ((end_x - start_x) / speed_x)

        # 判断输入值
        status = random.choice(["running", "waiting"]) if samples_list[i] == 0 else random.choice(["seal", "maintenance", "depleted"]) if samples_list[i] == 1 else None
        if status is None:
            raise ValueError("x必须是0或1")
        if status in ["running", "waiting"]: 
            # 正样本，运行状态
            if status == "running":
                power = random.uniform(50, 100)  # 初始电量范围保证运行状态下电量不会为0
                labels[i,0] = 1 # running=10000 
                for j in range(input_dim):
                    # 位置数据，根据上一时刻的位置和速度计算
                    if j == 0:
                        dataset[i, j, 0] = pos_x
                        dataset[i, j, 1] = pos_y
                        dataset[i, j, 3] = power
                        displacement = 0
                    else:
                        # 假设小车以恒定速度移动
                        dataset[i, j, 0] = dataset[i, j-1, 0] + speed_x
                        dataset[i, j, 1] = dataset[i, j-1, 1] + speed_y
                        # 计算位移距离
                        displacement = ((dataset[i, j, 0] - dataset[i, j-1, 0])**2 + (dataset[i, j, 1] - dataset[i, j-1, 1])**2)**0.5
                        power -= displacement * random.uniform(0.05, 0.1) 
                        dataset[i, j, 3] = power

            else: # 待料状态：小车到达工厂等待n秒装车后进行运行
                # 思路：选择一个小车到达工厂的时间去倒推小车之前的位置和电量，顺推小车未来的位置和电量
                power = random.uniform(50, 100)  # 初始电量范围保证运行状态下电量不会为0
                labels[i,1] = 1 # waiting = 01000
                stop_time = random.randint(0, input_dim - 1)  # 随机选择一个时刻进入待料状态
                for j in range(input_dim):
                    if j < stop_time:
                        dataset[i, j, 0] = end_x - (stop_time - j) * speed_x
                        dataset[i, j, 1] = end_y - (stop_time - j) * speed_y
                        displacement = ((dataset[i, stop_time, 0] - dataset[i, j, 0])**2 + (dataset[i, stop_time, 1] - dataset[i, j, 1])**2)**0.5
                        power -= displacement * random.uniform(0.05, 0.1)
                        dataset[i, j, 3] = power 
                    else: 
                        if j > stop_time + waiting_time:
                            dataset[i, j, 0] = dataset[i, j-1, 0] + speed_x
                            dataset[i, j, 1] = dataset[i, j-1, 1] + speed_y
                            displacement = ((dataset[i, stop_time, 0] - dataset[i, j, 0])**2 + (dataset[i, stop_time, 1] - dataset[i, j, 1])**2)**0.5
                            power -= displacement * random.uniform(0.05, 0.1)
                            dataset[i, j, 3] = power
                        else:
                            dataset[i, j, 0] = end_x
                            dataset[i, j, 1] = end_y
                            dataset[i, j, 3] = power
                   
        # 负样本，错误状态
        else:
            control_or_impersonation = random.choice(["control"]) if random.random() <= per_control else random.choice(["Impersonation"]) # 小车是否为冒充还是控制
            if control_or_impersonation == "Impersonation":
                device_list[-1] = "none" # 由于没用数据库，判定当设备指纹=none的时候设备遭到冒充
            
            # 封存状态
            if status == "seal":
                power = random.uniform(1, 100)  # 初始电量
                labels[i,2] = 1 # seal=00100
                # 封存状态下，位置数据不再变化，电量消耗也不再变化
                for j in range(input_dim):
                    dataset[i, j, 0] = pos_x
                    dataset[i, j, 1] = pos_y
                    dataset[i, j, 3] = power
            
            # 检修状态:小车在路上坏了，停止移动，某个随机时间停到某个地点
            elif status == "maintenance":
                power = random.uniform(50, 100)  # 初始电量范围是大于2，保证检修状态下电量不会为0
                labels[i,3] = 1 # waiting=00010
                # 思路： 随机一个时间和初始位置就是小车坏的位置和时间，去倒推之前之前的位置和电量
                stop_time = random.randint(1, input_dim - 1)  # 随机一个时刻停止移动
                for j in range(input_dim):
                    if j < stop_time:
                        dataset[i, j, 0] = pos_x - (stop_time - j) * speed_x
                        dataset[i, j, 1] = pos_y - (stop_time - j) * speed_y
                        displacement = ((dataset[i, stop_time, 0] - dataset[i, j, 0])**2 + (dataset[i, stop_time, 1] - dataset[i, j, 1])**2)**0.5
                        # 电量消耗数据，与位移距离成正比
                        power -= displacement * random.uniform(0.05, 0.1)         
                        dataset[i, j, 3] = power
                    else:
                        # 小车停止移动，位置和电量不再变化
                        dataset[i, j, 0] = pos_x
                        dataset[i, j, 1] = pos_y
                        dataset[i, j, 3] = power
            
            # 没电状态
            elif status == "depleted":
                labels[i,4] = 1 # 没电状态label=00001
                consume_power = [] # 消耗电量
                # 没电状态下，小车在某个时刻停止移动，且电量为0
                stop_time = random.randint(1, input_dim - 1)  # 随机选择一个时刻停止移动
                for j in range(input_dim):
                    if j <= stop_time:
                        # 小车移动
                        if j == 0:
                            dataset[i, j, 0] = pos_x
                            dataset[i, j, 1] = pos_y
                        else:
                            dataset[i, j, 0] = dataset[i, j-1, 0] + speed_x
                            dataset[i, j, 1] = dataset[i, j-1, 1] + speed_y
                            # 计算位移距离
                            displacement = ((dataset[i, j, 0] - dataset[i, j-1, 0])**2 + (dataset[i, j, 1] - dataset[i, j-1, 1])**2)**0.5
                            # 存入电量消耗
                            consume_power.append(displacement * random.uniform(0.05, 0.1)) 
                    else:
                        # 小车停止移动，电量为0
                        dataset[i, j, 0] = dataset[i, stop_time, 0]
                        dataset[i, j, 1] = dataset[i, stop_time, 1]
                        dataset[i, j, 3] = 0
                power = sum(consume_power)
                dataset[i, 0, 3] = power # 重新存入初始电量
                for j in range(stop_time):
                     power -= consume_power[j]
                     dataset[i,j+1,3] = max(0,power)
    
    dataset_ = torch.tensor(dataset, dtype=torch.float32)
    labels_ = torch.tensor(labels, dtype=torch.float32)
    print("...generate_car_data run Finished...")
    return dataset_, labels_, device_list


# 生成0和1的数组:0表示正样本，1表示负样本
def generate_samples_list(num_zeros, num_ones):
    samples_list = [0] * num_zeros + [1] * num_ones
    random.shuffle(samples_list)
    return samples_list

# 训练模型
def train_model(train_samples_list):
    # 初始化
    lstm = LSTM(input_size=4, hidden_size=64, num_layers=3, pred_output_size=4, clas_output_size=5)
    max_epochs = 10000
    optimizer = optim.Adam(lstm.parameters(), lr=1e-2, weight_decay=1e-5)

    # train_x, train_y, _ = generate_car_data(train_samples_list)
    # torch.save(train_x, './package/model_and_data/train_x.pt')
    # torch.save(train_y, './package/model_and_data/train_y.pt')
    train_x = torch.load('./package/model_and_data/train_x.pt')
    train_y = torch.load('./package/model_and_data/train_y.pt')
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检查 CUDA 可用性
    lstm = lstm.to(device_type)
    train_x = train_x.to(device_type)
    train_y = train_y.to(device_type)


    train_y_pred = train_x[:, -1, :]  # 预测任务标签
    train_y_clas = train_y  # 分类任务标签
    train_x = train_x[:, :-1, :]

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
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    return lstm

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

    return pred_labels


def authenticate_device(lstm, test_samples_list): #[1x4] 1000 0100 0010 0001
    
    test_x, test_y, _ = generate_car_data(test_samples_list)
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    lstm = lstm.to(device_type)
    test_x = test_x.to(device_type)
    test_y = test_y.to(device_type)
    pred_list=[]
    # 测试
    pred_labels = test_model(lstm, test_x, test_y)
    for i in range(pred_labels.size(0)):
        single_pred_label = pred_labels[i]
        label_index = torch.argmax(single_pred_label).item()   
        if label_index == 0 or label_index == 1:
            pred_list.append(0)
        else:
            pred_list.append(1)
    return pred_list


if __name__ == "__main__":

    # 模型训练(如果已经有训练好的模型可以注释这一段，后面直接加载训练好的模型即可)
    train_samples_list = generate_samples_list(10000,5000)
    lstm = train_model(train_samples_list)
    torch.save(lstm, './package/model_and_data/trainedmodel.pt')
    print("...Training Finished...")

    # 生成测试样本：0和1组成的数组
    test_samples_list = generate_samples_list(10, 5)
    # 加载训练好的模型
    lstm = torch.load('./package/model_and_data/trainedmodel.pt')  
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    lstm = lstm.to(device_type)

    # 模型验证
    pred_list = authenticate_device(lstm, test_samples_list)
    print(f"原始输入：{test_samples_list}")
    print(f"预测输出：{pred_list}")
