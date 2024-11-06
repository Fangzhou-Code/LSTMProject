import hashlib
import platform
import socket
import random
import uuid
from typing import Dict
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

# 根据设备指纹生成RSA密钥对
def generate_rsa_keypair_from_fingerprint(fingerprint: str):
    """
    使用设备指纹生成公私钥对
    """
    # 将指纹转换为整数种子
    seed = int(fingerprint, 16)
    random.seed(seed)

    # 生成RSA密钥对
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_key_pem.decode(), public_key_pem.decode()

# 生成设备指纹
def generate_device_fingerprint(device_info: Dict[str, str]) -> str:
    """
    生成设备指纹
    """
    info_string = ''.join([f"{key}:{value};" for key, value in device_info.items()])
    fingerprint = hashlib.sha256(info_string.encode()).hexdigest()
    return fingerprint

# 生成无人车数据
def generate_vehicle_data():
    """
    生成无人车的静态和动态数据，并生成设备指纹和RSA密钥对
    """
    # 根据时间戳生成唯一的 id
    vehicle_id = str(uuid.uuid1())

    # 生成名字和制造厂商
    name = 'car_' + str(random.randint(1, 999))
    manufacturer = 'manufacturer_' + str(random.randint(1, 999))

    # 生成静态信息
    static_info = {
        'id': vehicle_id,
        'name': name,
        'manufacturer': manufacturer,
        'device_type': 'car',
        'warranty_period': 12,  # 单位月
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "ip_address": socket.gethostbyname(socket.gethostname()),
        "mac_address": ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 2*6, 2)][::-1])
    }

    # 生成动态信息
    dynamic_info = {
        'position': (random.uniform(0, 100), random.uniform(0, 100)),
        'speed': random.uniform(0, 10),
        'power': random.randint(0, 100),
        'route': random.choice([1, 2, 3]),  # 选择路线
        'permissions': random.sample(['admin', 'operator', 'viewer'], random.randint(1, 3)),
        'frequence': 10
    }

    # 生成设备指纹
    fingerprint = generate_device_fingerprint(static_info)
    
    # 生成RSA密钥对
    private_key, public_key = generate_rsa_keypair_from_fingerprint(fingerprint)

    # 融合所有属性
    all_data = {
        'static_info': static_info,
        'dynamic_info': dynamic_info,
        'fingerprint': fingerprint,
        'private_key': private_key,
        'public_key': public_key
    }

    return all_data

# 获取特定属性
def get_attribute(data, section, attribute):
    """
    从生成的数据中获取特定属性
    """
    if section == 'static':
        return data['static_info'].get(attribute)
    elif section == 'dynamic':
        return data['dynamic_info'].get(attribute)
    elif section == 'fingerprint':
        return data.get('fingerprint') if attribute == 'fingerprint' else None
    elif section == 'private_key':
        return data.get('private_key') if attribute == 'private_key' else None
    elif section == 'public_key':
        return data.get('public_key') if attribute == 'public_key' else None
    else:
        print("该属性不存在")
        return None
        

# 修改特定属性
def set_attribute(data, section, attribute, value):
    """
    修改生成的数据中的特定属性
    """
    if section == 'static':
        data['static_info'][attribute] = value
    elif section == 'dynamic':
        if attribute == 'position':
            # 对于动态属性中的位置信息，特别处理
            current_position = data['dynamic_info'][attribute]
            new_position = (value, current_position[1])
            data['dynamic_info'][attribute] = new_position
        else:
            data['dynamic_info'][attribute] = value

# 测试获取和修改特定属性
if __name__ == "__main__":
    all_data = generate_vehicle_data()

    # 获取特定属性
    static_attribute = get_attribute(all_data, 'static', 'id')
    print("Static ID before modification:", static_attribute)

    dynamic_info = get_attribute(all_data, 'dynamic', 'position')
    print("Dynamic Position before modification:", dynamic_info)

    dynamic_attribute = get_attribute(all_data, 'dynamic', 'speed')
    print("Dynamic Speed before modification:", dynamic_attribute)

    # 获取指纹属性
    fingerprint = get_attribute(all_data, 'fingerprint', 'fingerprint')
    print("Fingerprint:", fingerprint, "type", type(fingerprint))

    # 获取私钥属性
    private_key = get_attribute(all_data, 'private_key', 'private_key')
    print("Private Key:", private_key)

    # 获取公钥属性
    public_key = get_attribute(all_data, 'public_key', 'public_key')
    print("Public Key:", public_key)

    # 修改特定属性
    set_attribute(all_data, 'static', 'id', 'new_id_value')
    new_x_value = 50.0
    dynamic_position = get_attribute(all_data, 'dynamic', 'position')
    new_dynamic_position = (new_x_value, dynamic_position[1])
    set_attribute(all_data, 'dynamic', 'position', new_dynamic_position)
    set_attribute(all_data, 'dynamic', 'speed', 20.5)

    # 再次获取特定属性，确认修改
    static_attribute = get_attribute(all_data, 'static', 'id')
    print("Static ID after modification:", static_attribute)

    dynamic_info = get_attribute(all_data, 'dynamic', 'position')
    print("Dynamic Position after modification:", dynamic_info)

    dynamic_attribute = get_attribute(all_data, 'dynamic', 'speed')
    print("Dynamic Speed after modification:", dynamic_attribute)