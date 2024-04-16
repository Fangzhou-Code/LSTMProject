# LSTM 小车身份认证

## 1. 定义行为模式

正常的小车的行为模式和错误小车的行为模式；

## 2. 提升性能

* 增加属性数量
  
目前的属性是：小车的速度、直线行驶、电量消耗、运行时间 ${x_{t}, y_{t}, z_{t}}$，属性个数是`INPUT_SIZE = 3`

* 增加LSTM层数

当前是

```python
HIDDEN_SIZE = 10 
NUM_LAYERS = 1
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
