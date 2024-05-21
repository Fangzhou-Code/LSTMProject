# LSTM 小车身份认证

## 1. 定义行为模式

正常的小车的行为模式和错误小车的行为模式；

思路：我们将模型的输出分为两类——预测值和分类概率分布，两者同时决定设备身份验证结果

预测值：我们每隔k时刻取${t_{1},t_{2},t_{3},...,t_{n}}$时刻的小车数据时空位置和电量。我们用$t_{1},t_{2},t_{3},...,t_{n-1}$时刻的数据进行训练，预测第$t_{n}$时刻的数据

分类概率：我们将四种状态进行one-hot编码：运行 1000、封存 0100、检修 0010、待料 0001 <br>
要求:<br>
正样本状态是运行状态;<br>
负样本产生三种错误状态: 封存 、检修 、待料三种错误状态.同时三种错误状态对应不同的错误数据，封存状态数据是不同时刻位置状态和初始状态相同，检修状态是位移到某个随即时刻后不再移动，有电量，待料状态是某个时刻后不再移动，电量为0.

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

## 4. 生成公私钥对

```python
class DeviceAuthentication:
    def __init__(self, device_id, manufacturer):
        self.device_id = device_id
        self.manufacturer = manufacturer
    def authenticate_device(self):
        # 假设这是一个简单的身份验证逻辑，定义小车的行为模式
        return True
    def issue_credentials(self):
        # 生成随机的凭证和密钥对
        credential = str(uuid.uuid4())
        public_key, private_key = self.generate_key_pair()
        return credential, public_key, private_key
    def generate_key_pair(self):
        # 生成RSA公私钥对
        pubkey, privkey = rsa.newkeys(2048)
         # 将公私钥对保存到文件中
        with open('key_pair/public_key.pem', 'wb') as public_key_file:
            public_key_file.write(pubkey.save_pkcs1())
        with open('key_pair/private_key.pem', 'wb') as private_key_file:
            private_key_file.write(privkey.save_pkcs1())
        return pubkey.save_pkcs1(), privkey.save_pkcs1()
    def load_keys():
        # 加载已保存的公私钥对
        with open('key_pair/public_key.pem', 'rb') as public_key_file:
            pubkey = rsa.PublicKey.load_pkcs1(public_key_file.read())
        with open('key_pair/private_key.pem', 'rb') as private_key_file:
            privkey = rsa.PrivateKey.load_pkcs1(private_key_file.read())
        return pubkey, privkey
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

[device.py](./device.py)

**固定属性:**

* ID
* 设备名字
* 制造商
* 类型
* 位置（xyz）
* 速度
* 电量
* 移动范围
* 权限
* 保修信息

**思路**

* 生成数据函数：根据固定属性生成数据
* 发送数据函数：然后发送数据到服务器
* 获取数据函数：正常从api获取数据，但是目前没用api应该直接获取数据，因此这部分没有对齐


4. 和别的方案进行对比（调研）

