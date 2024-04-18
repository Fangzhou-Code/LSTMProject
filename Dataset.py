import numpy as np
import random
import torch

def generate_car_data(per_positive):
    num_samples=10000
    input_dim=10
    dataset = np.zeros((num_samples, input_dim, 3))
    labels = np.zeros((num_samples,4),dtype=int)

    for i in range(num_samples):
        # 随机初始化小车的位置和状态
        pos_x = random.uniform(0, 10)
        pos_y = random.uniform(0, 10)
        status = "running" if random.random() < per_positive else random.choice(["seal", "maintenance", "waiting"])
        
        # 正样本，运行状态
        if status == "running": 
            power = random.uniform(2, 100)  # 初始电量范围是大于2，保证运行状态下电量不会为0
            labels[i,0] = 1 # 运行状态label=1000 
            for j in range(input_dim):
                # 位置数据，根据上一时刻的位置和速度计算
                if j == 0:
                    dataset[i, j, 0] = pos_x
                    dataset[i, j, 1] = pos_y
                else:
                    # 假设小车以恒定速度移动
                    dataset[i, j, 0] = dataset[i, j-1, 0] + random.uniform(0, 0.5)
                    dataset[i, j, 1] = dataset[i, j-1, 1] + random.uniform(0, 0.5)
                # 计算位移距离
                if j > 0:
                    displacement = ((dataset[i, j, 0] - dataset[i, j-1, 0])**2 + (dataset[i, j, 1] - dataset[i, j-1, 1])**2)**0.5
                else:
                    displacement = 0
                # 电量消耗数据，与位移距离成正比
                power -= displacement * random.uniform(0.05, 0.1) 
                dataset[i, j, 2] = power
        
        # 负样本，错误状态
        else:
            # 封存状态
            if status == "seal":
                power = random.uniform(1, 100)  # 初始电量
                labels[i,1] = 1 # 封存状态label=0100
                # 封存状态下，位置数据不再变化，电量消耗也不再变化
                for j in range(input_dim):
                    dataset[i, j, 0] = pos_x
                    dataset[i, j, 1] = pos_y
                    dataset[i, j, 2] = power
            
            # 检修状态
            elif status == "maintenance":
                power = random.uniform(2, 100)  # 初始电量范围是大于2，保证检修状态下电量不会为0
                labels[i,2] = 1 # 检修状态label=0010
                # 检修状态下，小车在某个随机时刻停止移动，但电量消耗仍在继续
                stop_time = random.randint(1, input_dim - 1)  # 随机选择一个时刻停止移动
                for j in range(input_dim):
                    if j < stop_time:
                        # 小车移动
                        if j == 0:
                            dataset[i, j, 0] = pos_x
                            dataset[i, j, 1] = pos_y
                            displacement = 0
                        else:
                            dataset[i, j, 0] = dataset[i, j-1, 0] + random.uniform(0, 0.5)
                            dataset[i, j, 1] = dataset[i, j-1, 1] + random.uniform(0, 0.5)
                            displacement = ((dataset[i, j, 0] - dataset[i, j-1, 0])**2 + (dataset[i, j, 1] - dataset[i, j-1, 1])**2)**0.5
                        # 电量消耗数据，与位移距离成正比
                        power -= displacement * random.uniform(0.05, 0.1) 
                        dataset[i, j, 2] = power
                    else:
                        # 小车停止移动，位置和电量不再变化
                        dataset[i, j, 0] = dataset[i, stop_time-1, 0]
                        dataset[i, j, 1] = dataset[i, stop_time-1, 1]
                        dataset[i, j, 2] = dataset[i, stop_time-1, 2]
            
            # 待料状态
            elif status == "waiting":
                labels[i,3] = 1 # 待料状态label=0001
                consume_power = [] # 消耗电量
                # 待料状态下，小车在某个时刻停止移动，且电量为0
                stop_time = random.randint(1, input_dim - 1)  # 随机选择一个时刻停止移动
                for j in range(input_dim):
                    if j <= stop_time:
                        # 小车移动
                        if j == 0:
                            dataset[i, j, 0] = pos_x
                            dataset[i, j, 1] = pos_y
                            displacement = 0
                        else:
                            dataset[i, j, 0] = dataset[i, j-1, 0] + random.uniform(0, 0.5)
                            dataset[i, j, 1] = dataset[i, j-1, 1] + random.uniform(0, 0.5)
                            # 计算位移距离
                            displacement = ((dataset[i, j, 0] - dataset[i, j-1, 0])**2 + (dataset[i, j, 1] - dataset[i, j-1, 1])**2)**0.5
                            # 存入电量消耗
                            consume_power.append(displacement * random.uniform(0.05, 0.1)) 
                    else:
                        # 小车停止移动，电量为0
                        dataset[i, j, 0] = dataset[i, stop_time, 0]
                        dataset[i, j, 1] = dataset[i, stop_time, 1]
                        dataset[i, j, 2] = 0
                power = sum(consume_power)
                dataset[i, 0, 2] = power # 重新存入初始电量
                for j in range(stop_time):
                     power -= consume_power[j]
                     dataset[i,j+1,2] = max(0,power)
    dataset_ = torch.tensor(dataset, dtype=torch.float32)
    labels_ = torch.tensor(labels, dtype=torch.float32)
    print("...generate_car_data run Finished...")
    return dataset_, labels_



if __name__=="__main__":
    # 生成训练集
    traindataset,trainlabels = generate_car_data(per_positive=0.7)
    torch.save(traindataset, 'Dataset/traindataset.pt')
    torch.save(trainlabels, 'Dataset/trainlabels.pt') 
    print("...train Create Finished...")

    # 生成测试集
    testdataset,testlabels = generate_car_data(per_positive=0.7)
    torch.save(testdataset, 'Dataset/testdataset.pt')
    torch.save(testlabels, 'Dataset/testlabels.pt')
    print("...test Create Finished...")


    # 输出运行状态样本案例
    print("Running status sample:")
    running_sample_idx = np.argmax(trainlabels[:, 0].numpy())  # 找到第一个运行状态样本的索引
    print("Dataset:")
    print(traindataset[running_sample_idx])
    print("Labels:")
    print(trainlabels[running_sample_idx])

    # 输出封存状态样本案例
    print("\nSeal status sample:")
    seal_sample_idx = np.argmax(trainlabels[:, 1].numpy())  # 找到第一个封存状态样本的索引
    print("Dataset:")
    print(traindataset[seal_sample_idx])
    print("Labels:")
    print(trainlabels[seal_sample_idx])

    # 输出检修状态样本案例
    print("\nMaintenance status sample:")
    maintenance_sample_idx = np.argmax(trainlabels[:, 2].numpy())  # 找到第一个检修状态样本的索引
    print("Dataset:")
    print(traindataset[maintenance_sample_idx])
    print("Labels:")
    print(trainlabels[maintenance_sample_idx])

    # 输出待料状态样本案例
    print("\nWaiting status sample:")
    waiting_sample_idx = np.argmax(trainlabels[:, 3].numpy())  # 找到第一个待料状态样本的索引
    print("Dataset:")
    print(traindataset[waiting_sample_idx])
    print("Labels:")
    print(trainlabels[waiting_sample_idx])