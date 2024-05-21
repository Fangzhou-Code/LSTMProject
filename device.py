import requests
import random
import time
import uuid
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import numpy as np
import random
import torch


# 服务器地址
API_URL = 'http://localhost:5000/vehicles'

# 生成RSA密钥对
def generate_rsa_keypair():
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




# 生成无人车数据
def generate_vehicle_data():
    # 根据时间戳生成唯一的 id
    vehicle_id = str(uuid.uuid1())

    # 生成名字和制造厂商
    name = 'car_' + str(random.randint(100))
    manufacturer = 'manufacturer_' +  str(random.randint(100))

    # 生成位置、速度、电量、移动路线等随机属性
    position = (random.uniform(0, 100), random.uniform(0, 100))
    speed = random.uniform(0, 10)
    battery_level = random.randint(0, 100)
    route = random.choice([1, 2, 3])
    permissions = ['admin', 'operator', 'viewer']

    # 生成RSA密钥对
    private_key, public_key = generate_rsa_keypair()

    # 构建数据
    data = {
        'id': vehicle_id,
        'name': name,
        'manufacturer': manufacturer,
        'device_type': 'car',
        'position': position,
        'speed': speed,
        'power': battery_level,
        'route': route, # 选择路线
        'permissions': random.sample(permissions, random.randint(1, len(permissions))),
        'warranty_period': 12, # 单位月
        'public_key': public_key,
        'private_key': private_key
    }
    return data

# 发送无人车数据到服务器
def send_vehicle_data(data):
    response = requests.post(API_URL, json=data)
    if response.status_code == 201:
        print("Vehicle data sent successfully.")
    else:
        print("Failed to send vehicle data:", response.status_code)

# 获取全部小车数据
def get_all_vehicles():
    response = requests.get(API_URL)
    if response.status_code == 200:
        vehicles = response.json()
        print("All vehicles:")
        for vehicle_id, vehicle_info in vehicles.items():
            print(f"ID: {vehicle_id}, Info: {vehicle_info}")
    else:
        print("Failed to get vehicles:", response.status_code)

# 获取指定小车数据
def get_vehicle_by_id(vehicle_id, API_URL):
    response = requests.get(API_URL)
    if response.status_code == 200:
        vehicle_data = response.json()
        print("Vehicle data:")
        print(vehicle_data)
    elif response.status_code == 404:
        print("Vehicle not found.")
    else:
        print("Failed to get vehicle data:", response.status_code)

if __name__ == '__main__':
    while True:
        new_vehicle_data = generate_vehicle_data()  # 生成新的无人车数据
        send_vehicle_data(new_vehicle_data)  # 发送数据到服务器
        time.sleep(5)  # 模拟每隔5秒发送一次数据
        # get_all_vehicles
        # get_vehicle_by_id(vehicle_id)