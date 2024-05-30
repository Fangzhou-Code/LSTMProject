# LSTM 小车身份认证

## 1. 定义行为模式

正常的小车的行为模式和错误小车的行为模式；

思路：我们将模型的输出分为两类——预测值和分类概率分布，两者同时决定设备身份验证结果

预测值：我们每隔k时刻取${t_{1},t_{2},t_{3},...,t_{n}}$时刻的小车数据时空位置和电量。我们用$t_{1},t_{2},t_{3},...,t_{n-1}$时刻的数据进行训练，预测第$t_{n}$时刻的数据

分类概率：我们将四种状态进行one-hot编码：

* 运行 10000
* 待料 01000
* 封存 00100
* 检修 00010
* 没电 00001 
  
要求:

正样本状态是运行状态和待料状态(等待6秒)

负样本产生三种错误状态: 封存 、检修 、没电三种错误状态.同时三种错误状态对应不同的错误数据，封存状态数据是不同时刻位置状态和初始状态相同，检修状态是位移到某个随即时刻后不再移动，有电量，没电状态是某个时刻后不再移动，电量为0.

## 2. 提升性能

* 增加属性数量
  
目前的属性是：小车的速度、直线行驶、电量消耗、运行时间 ${x_{t}, y_{t}, z_{t}}$，属性个数是`INPUT_SIZE = 3`

* 增加LSTM层数

当前是

```python
HIDDEN_SIZE = 32 
NUM_LAYERS = 3
```

* 对比试验

## 3. 测试时延

```python
import time
start_time = time.time()  # 记录开始时间
some_function()  # 调用函数
end_time = time.time()  # 记录结束时间
execution_time = end_time - start_time  # 计算运行时间
print(f"函数运行时间: {execution_time}秒")
```

## 4. 代码介绍

### Dataset.py

#### 代码介绍

本文档介绍了一个用于生成小车运行数据集的Python代码。代码主要用于模拟小车在不同状态下的运行轨迹和电量消耗，并生成相应的数据集以供训练和测试使用。

#### 主要功能

代码的主要功能包括：

1. 判断某点是否在目标点附近的 `is_nearby` 函数。
2. 生成小车数据集的 `generate_car_data` 函数。
3. 保存训练集和测试集数据。
4. 输出不同状态下的小车数据样本。


#### Device.py

本项目的主要目标是生成包含静态和动态信息的无人车数据，并基于设备指纹生成公私钥对。以下是项目的详细介绍：

主要功能

* 生成设备指纹
* 生成RSA密钥对
* 生成无人车数据
* 获取特定属性
* 修改特定属性

属性分类：

静态属性、动态属性、设备指纹、公私钥

* 静态属性：
  
| 属性名            | 类型   | 描述                          |
| ----------------- | ------ | ----------------------------- |
| `id`              | 字符串 | 唯一标识符，由UUID生成         |
| `name`            | 字符串 | 无人车的名称                   |
| `manufacturer`    | 字符串 | 制造商名称                     |
| `device_type`     | 字符串 | 设备类型，例如：`car`          |
| `warranty_period` | 整数   | 保修期，以月为单位             |
| `os`              | 字符串 | 操作系统                       |
| `os_version`      | 字符串 | 操作系统版本                   |
| `machine`         | 字符串 | 机器类型（硬件信息）           |
| `processor`       | 字符串 | 处理器信息                     |
| `hostname`        | 字符串 | 主机名                         |
| `ip_address`      | 字符串 | IP 地址                        |
| `mac_address`     | 字符串 | MAC 地址                       |

示例：
```python
{
    "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "name": "car_123",
    "manufacturer": "manufacturer_456",
    "device_type": "car",
    "warranty_period": 12,
    "os": "Linux",
    "os_version": "5.4.0-74-generic",
    "machine": "x86_64",
    "processor": "Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz",
    "hostname": "car-hostname",
    "ip_address": "192.168.1.10",
    "mac_address": "00:1A:2B:3C:4D:5E"
}
```

* 动态属性
  
| 属性名       | 类型             | 描述                          |
| ------------ | ---------------- | ----------------------------- |
| `position`   | 元组 (浮点数, 浮点数) | 无人车的当前位置               |
| `speed`      | 浮点数           | 无人车的速度                   |
| `power`      | 整数             | 无人车的电量                   |
| `route`      | 整数             | 无人车当前的路线编号           |
| `permissions`| 列表（字符串）   | 无人车的权限，例如：`['admin', 'operator']` |
| `frequence`  | 整数             | 无人车数据更新的频率，以秒为单位 |

示例：

```python
{
    "position": [52.3765, 4.8945],
    "speed": 8.5,
    "power": 85,
    "route": 2,
    "permissions": ["admin", "viewer"],
    "frequence": 10
}
```

全部属性完整示例：

```python
{
    "static_info": {
        "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "name": "car_123",
        "manufacturer": "manufacturer_456",
        "device_type": "car",
        "warranty_period": 12,
        "os": "Linux",
        "os_version": "5.4.0-74-generic",
        "machine": "x86_64",
        "processor": "Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz",
        "hostname": "car-hostname",
        "ip_address": "192.168.1.10",
        "mac_address": "00:1A:2B:3C:4D:5E"
    },
    "dynamic_info": {
        "position": [52.3765, 4.8945],
        "speed": 8.5,
        "power": 85,
        "route": 2,
        "permissions": ["admin", "viewer"],
        "frequence": 10
    },
    "fingerprint": "a9b7ba70783b617e9998dc4dd82eb3c5",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIE... (省略) ...QAB\n-----END PRIVATE KEY-----",
    "public_key": "-----BEGIN PUBLIC KEY-----\nMIIBIj... (省略) ...wIDAQAB\n-----END PUBLIC KEY-----"
}

```





## 5. 优化

1. 定义正常模式

定义正常运行路线:[route.py](./route.py)

2. 对比实验

* 测试集损失值、准确率的变化：折线图
* 更新和不更新凭证设备身份验证的准确率：柱状图对比
* 篡改时长：[1-10] （暂时不做）
* 篡改时长位置: [2-5,7-10] （暂时不做）

3. 小车属性固定

静态属性、动态属性、公私钥、设备指纹：[device.py](./device.py)


**注意:[run.py](./run.py) 主函数可能存在问题，模型的测试和训练移动到了[train_test_model](./train_test_model.py)**




## 6. 对比方案

和设备指纹进行对比：[devicefinger_test.py](./devicefinger_test.py)
