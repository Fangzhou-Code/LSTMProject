import numpy as np
import random
import torch
import route
import math
import device
import pickle




def generate_car_data(num_samples, input_dim, per_positive):
    # 初始化
    dataset = np.zeros((num_samples, input_dim, 3))
    labels = np.zeros((num_samples,5),dtype=int)
    waiting_time = 6 # 待料状态下需要等待6秒
    device_list = []
    per_control = 0.5 # 控制数量占全部负样本的比例
    print("开始生成样本")
    for i in range(num_samples):
        print(f"生成样本 {i + 1}/{num_samples}")
        # 随机路线
        route_number = random.randint(1,3)
        start_point, end_point = route.get_route_coordinates(route_number)
        # print("Route", route_number, "coordinates:")
        # print("Start point:", start_point)
        # print("End point:", end_point)
        end_x = end_point[0]
        end_y = end_point[1]
        start_x = start_point[0]
        start_y = start_point[1]
        # 随机初始化小车的静态属性和位置和状态
        all_data = device.generate_vehicle_data()
        fingerpoint = device.get_attribute(all_data, 'fingerprint', 'fingerprint')
        device_list.append(fingerpoint)
        distance = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5 
        speed_x = (end_x - start_x) / distance 
        speed_y = (end_y - start_y) / distance
        pos_x = random.uniform(start_x, end_x)
        pos_y = start_y + speed_y * ((end_x - start_x) / speed_x)

        status = random.choice(["running", "waiting"]) if random.random() < per_positive else random.choice(["seal", "maintenance", "depleted"])
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
                        dataset[i, j, 2] = power
                        displacement = 0
                    else:
                        # 假设小车以恒定速度移动
                        dataset[i, j, 0] = dataset[i, j-1, 0] + speed_x
                        dataset[i, j, 1] = dataset[i, j-1, 1] + speed_y
                        # 计算位移距离
                        displacement = ((dataset[i, j, 0] - dataset[i, j-1, 0])**2 + (dataset[i, j, 1] - dataset[i, j-1, 1])**2)**0.5
                        power -= displacement * random.uniform(0.05, 0.1) 
                        dataset[i, j, 2] = power

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
                        dataset[i, j, 2] = power 
                    else: 
                        if j > stop_time + waiting_time:
                            dataset[i, j, 0] = dataset[i, j-1, 0] + speed_x
                            dataset[i, j, 1] = dataset[i, j-1, 1] + speed_y
                            displacement = ((dataset[i, stop_time, 0] - dataset[i, j, 0])**2 + (dataset[i, stop_time, 1] - dataset[i, j, 1])**2)**0.5
                            power -= displacement * random.uniform(0.05, 0.1)
                            dataset[i, j, 2] = power
                        else:
                            dataset[i, j, 0] = end_x
                            dataset[i, j, 1] = end_y
                            dataset[i, j, 2] = power
                   
        # 负样本，错误状态
        else:
            control_or_impersonation = random.choice(["control"]) if random.random() <= per_control else random.choice(["Impersonation"])
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
                    dataset[i, j, 2] = power
            
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
                        dataset[i, j, 2] = power
                    else:
                        # 小车停止移动，位置和电量不再变化
                        dataset[i, j, 0] = pos_x
                        dataset[i, j, 1] = pos_y
                        dataset[i, j, 2] = power
            
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
                        dataset[i, j, 2] = 0
                power = sum(consume_power)
                dataset[i, 0, 2] = power # 重新存入初始电量
                for j in range(stop_time):
                     power -= consume_power[j]
                     dataset[i,j+1,2] = max(0,power)
    
    dataset_ = torch.tensor(dataset, dtype=torch.float32)
    labels_ = torch.tensor(labels, dtype=torch.float32)
    print("...generate_car_data run Finished...")
    return dataset_, labels_, device_list


def generate_forklift_data(num_samples, input_dim, per_positive):
    # 初始化
    dataset = np.zeros((num_samples, input_dim, 3))
    labels = np.zeros((num_samples,5),dtype=int)
    waiting_time = 6 # 待料状态下需要等待6秒
    device_list = []
    per_control = 0.5 # 控制数量占全部负样本的比例
    print("开始生成样本")
    for i in range(num_samples):
        print(f"生成样本 {i + 1}/{num_samples}")
        # 随机路线
        route_number = random.randint(4,8) 
        start_point, end_point = route.get_route_coordinates(route_number)
        # print("Route", route_number, "coordinates:")
        # print("Start point:", start_point)
        # print("End point:", end_point)
        end_x = end_point[0]
        end_y = end_point[1]
        start_x = start_point[0]
        start_y = start_point[1]
        # 随机初始化小车的静态属性和位置和状态
        all_data = device.generate_vehicle_data()
        fingerpoint = device.get_attribute(all_data, 'fingerprint', 'fingerprint')
        device_list.append(fingerpoint)
        distance = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5 
        speed_x = (end_x - start_x) / distance 
        speed_y = (end_y - start_y) / distance
        pos_x = random.uniform(start_x, end_x)
        pos_y = start_y + speed_y * ((end_x - start_x) / speed_x)

        status = random.choice(["running", "waiting"]) if random.random() < per_positive else random.choice(["seal", "maintenance", "depleted"])

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
                        dataset[i, j, 2] = power
                        displacement = 0
                    else:
                        # 假设小车以恒定速度移动
                        dataset[i, j, 0] = dataset[i, j-1, 0] + speed_x
                        dataset[i, j, 1] = dataset[i, j-1, 1] + speed_y
                        # 计算位移距离
                        displacement = ((dataset[i, j, 0] - dataset[i, j-1, 0])**2 + (dataset[i, j, 1] - dataset[i, j-1, 1])**2)**0.5
                        power -= displacement * random.uniform(1, 2) 
                        dataset[i, j, 2] = power

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
                        power -= displacement * random.uniform(1, 2)
                        dataset[i, j, 2] = power 
                    else: 
                        if j > stop_time + waiting_time:
                            dataset[i, j, 0] = dataset[i, j-1, 0] + speed_x
                            dataset[i, j, 1] = dataset[i, j-1, 1] + speed_y
                            displacement = ((dataset[i, stop_time, 0] - dataset[i, j, 0])**2 + (dataset[i, stop_time, 1] - dataset[i, j, 1])**2)**0.5
                            power -= displacement * random.uniform(1, 2)
                            dataset[i, j, 2] = power
                        else:
                            dataset[i, j, 0] = end_x
                            dataset[i, j, 1] = end_y
                            dataset[i, j, 2] = power
                   
        # 负样本，错误状态
        else:
            control_or_impersonation = random.choice(["control"]) if random.random() <= per_control else random.choice(["Impersonation"])
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
                    dataset[i, j, 2] = power
            
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
                        power -= displacement * random.uniform(1, 2)         
                        dataset[i, j, 2] = power
                    else:
                        # 小车停止移动，位置和电量不再变化
                        dataset[i, j, 0] = pos_x
                        dataset[i, j, 1] = pos_y
                        dataset[i, j, 2] = power
            
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
                            consume_power.append(displacement * random.uniform(1, 2)) 
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
    print("...generate_forklift_data run Finished...")
    return dataset_, labels_, device_list



def generate_uav_data(num_samples, input_dim, per_positive):
    # 初始化
    dataset = np.zeros((num_samples, input_dim, 3))
    labels = np.zeros((num_samples,5),dtype=int)
    waiting_time = 6 # 待料状态下需要等待6秒
    device_list = []
    per_control = 0.5 # 控制数量占全部负样本的比例
    print("开始生成样本")
    for i in range(num_samples):
        print(f"生成样本 {i + 1}/{num_samples}")
        # 随机路线
        route_number = random.randint(9,12)
        start_point, end_point = route.get_route_coordinates(route_number)
        start_x, start_y, start_z = start_point
        end_x, end_y, end_z = end_point
        # 随机初始化小车的静态属性和位置和状态
        all_data = device.generate_vehicle_data()
        fingerpoint = device.get_attribute(all_data, 'fingerprint', 'fingerprint')
        device_list.append(fingerpoint)
        distance = ((end_x - start_x)**2 + (end_y - start_y)**2 + (end_z - start_z))**0.5 
        speed_x = (end_x - start_x) / distance 
        speed_y = (end_y - start_y) / distance
        speed_z =  (end_z - start_z) / distance
        pos_x = random.uniform(start_x, end_x)
        pos_y = start_y + speed_y * ((end_x - start_x) / speed_x)
        pos_z = start_z + speed_z * ((end_x - start_x) / speed_x)

        status = random.choice(["running", "waiting"]) if random.random() < per_positive else random.choice(["seal", "maintenance", "depleted"])

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
                        dataset[i, j, 2] = pos_z
                        dataset[i, j, 3] = power
                        displacement = 0
                    else:
                        # 假设小车以恒定速度移动
                        dataset[i, j, 0] = dataset[i, j-1, 0] + speed_x
                        dataset[i, j, 1] = dataset[i, j-1, 1] + speed_y
                        dataset[i, j, 2] = dataset[i, j-1, 2] + speed_z
                        # 计算位移距离
                        displacement = ((dataset[i, j, 0] - dataset[i, j-1, 0])**2 + (dataset[i, j, 1] - dataset[i, j-1, 1])**2 + (dataset[i, j, 2] - dataset[i, j-1, 2])**2)**0.5
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
                        dataset[i, j, 2] = power 
                    else: 
                        if j > stop_time + waiting_time:
                            dataset[i, j, 0] = dataset[i, j-1, 0] + speed_x
                            dataset[i, j, 1] = dataset[i, j-1, 1] + speed_y
                            displacement = ((dataset[i, stop_time, 0] - dataset[i, j, 0])**2 + (dataset[i, stop_time, 1] - dataset[i, j, 1])**2)**0.5
                            power -= displacement * random.uniform(0.05, 0.1)
                            dataset[i, j, 2] = power
                        else:
                            dataset[i, j, 0] = end_x
                            dataset[i, j, 1] = end_y
                            dataset[i, j, 2] = power
                   
        # 负样本，错误状态
        else:
            control_or_impersonation = random.choice(["control"]) if random.random() <= per_control else random.choice(["Impersonation"])
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
                    dataset[i, j, 2] = power
            
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
                        dataset[i, j, 2] = power
                    else:
                        # 小车停止移动，位置和电量不再变化
                        dataset[i, j, 0] = pos_x
                        dataset[i, j, 1] = pos_y
                        dataset[i, j, 2] = power
            
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
                        dataset[i, j, 2] = 0
                power = sum(consume_power)
                dataset[i, 0, 2] = power # 重新存入初始电量
                for j in range(stop_time):
                     power -= consume_power[j]
                     dataset[i,j+1,2] = max(0,power)
    
    dataset_ = torch.tensor(dataset, dtype=torch.float32)
    labels_ = torch.tensor(labels, dtype=torch.float32)
    print("...generate_car_data run Finished...")
    return dataset_, labels_, device_list



if __name__=="__main__":
    # 生成训练集
    traindataset,trainlabels,traindevicelist = generate_car_data(num_samples=10000, input_dim=20, per_positive=0.6)
    torch.save(traindataset, 'Dataset/traindataset.pt')
    torch.save(trainlabels, 'Dataset/trainlabels.pt')
    # 保存list数组到本地文件
    with open('Dataset/traindevicelist.pkl', 'wb') as f:
        pickle.dump(traindevicelist, f)
    print("...train Create Finished...")

    # 生成测试集
    testdataset,testlabels,testdevicelist = generate_car_data(num_samples=1000, input_dim=20, per_positive=0.7)
    torch.save(testdataset, 'Dataset/testdataset.pt')
    torch.save(testlabels, 'Dataset/testlabels.pt') 
    # 保存list数组到本地文件
    with open('Dataset/testdevicelist.pkl', 'wb') as f:
        pickle.dump(testdevicelist, f)
    print("...test Create Finished...")


    # 输出运行状态样本案例
    print("Running status sample:")
    running_sample_idx = np.argmax(trainlabels[:, 0].numpy())  # 找到第一个运行状态样本的索引
    print("Dataset:")
    print(traindataset[running_sample_idx])
    print("Labels:")
    print(trainlabels[running_sample_idx])
    print("Fingerpoint:")
    print(traindevicelist[running_sample_idx])

    # 输出待料状态样本案例
    print("Waiting status sample:")
    waiting_sample_idx = np.argmax(trainlabels[:, 1].numpy())  # 找到第一个运行状态样本的索引
    print("Dataset:")
    print(traindataset[waiting_sample_idx])
    print("Labels:")
    print(trainlabels[waiting_sample_idx])
    print("Fingerpoint:")
    print(traindevicelist[running_sample_idx])

    # 输出封存状态样本案例
    print("\nSeal status sample:")
    seal_sample_idx = np.argmax(trainlabels[:, 2].numpy())  # 找到第一个封存状态样本的索引
    print("Dataset:")
    print(traindataset[seal_sample_idx])
    print("Labels:")
    print(trainlabels[seal_sample_idx])
    print("Fingerpoint:")
    print(traindevicelist[running_sample_idx])

    # 输出检修状态样本案例
    print("\nMaintenance status sample:")
    maintenance_sample_idx = np.argmax(trainlabels[:, 3].numpy())  # 找到第一个检修状态样本的索引
    print("Dataset:")
    print(traindataset[maintenance_sample_idx])
    print("Labels:")
    print(trainlabels[maintenance_sample_idx])
    print("Fingerpoint:")
    print(traindevicelist[running_sample_idx])

    # 输出没电状态样本案例
    print("\nDepleted status sample:")
    depleted_sample_idx = np.argmax(trainlabels[:, 4].numpy())  # 找到第一个停电状态样本的索引
    print("Dataset:")
    print(traindataset[depleted_sample_idx])
    print("Labels:")
    print(trainlabels[depleted_sample_idx])
    print("Fingerpoint:")
    print(traindevicelist[running_sample_idx])