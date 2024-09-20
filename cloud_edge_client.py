import time
import pickle
import torch
import socket
from multiprocessing import Process, Queue
from Dataset import generate_car_data
from Model import LSTM
import run

# 定义全局参数
INPUT_DIM = 10
TRAIN_EPOCHS = 400
SFT_EPOCHS = 100
PER_POSITIVE = 0.6
NUM_SAMPLES_TRAIN = 1000
NUM_SAMPLES_TEST = 100
INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 3
PRED_OUTPUT_SIZE = 3
CLAS_OUTPUT_SIZE = 5

# 禁用 MKL-DNN 以避免在 GPU 上运行时遇到问题
torch.backends.mkldnn.enabled = False

def send_data(conn, data):
    """通过socket发送数据，并先发送数据长度"""
    data_bytes = pickle.dumps(data)  # 序列化数据
    data_len = len(data_bytes)  # 数据长度
    conn.sendall(data_len.to_bytes(8, 'big'))  # 先发送8字节的数据长度
    conn.sendall(data_bytes)  # 发送实际数据 

def recv_data(conn):
    """通过socket接收数据，先接收数据长度再接收数据"""
    data_len_bytes = conn.recv(8)  # 先接收8字节的数据长度
    if not data_len_bytes:
        return None
    data_len = int.from_bytes(data_len_bytes, 'big')
    data_bytes = b''

    # 循环接收完整数据
    while len(data_bytes) < data_len:
        packet = conn.recv(min(4096, data_len - len(data_bytes)))
        if not packet:
            break
        data_bytes += packet

    return pickle.loads(data_bytes)  # 反序列化并返回数据

def save_data(train_x, train_y, device_list, prefix='train'):
    """保存数据集及设备列表到指定路径"""
    try:
        torch.save(train_x, f'Dataset/{prefix}dataset.pt')
        torch.save(train_y, f'Dataset/{prefix}labels.pt')
        with open(f'Dataset/{prefix}devicelist.pkl', 'wb') as f:
            pickle.dump(device_list, f)
        print(f"...{prefix} dataset Create Finished...")
    except Exception as e:
        print(f"Error saving {prefix} data: {e}")

def test_and_authenticate_model(lstm, test_x, test_y, device_id, manufacturer):
    """加载模型并进行测试和设备认证"""
    lstm = run.load_model()
    device = run.DeviceAuthentication(device_id, manufacturer)
    test_loss, test_acc, pred_y_pred, pred_y_clas, pred_labels = run.test_model(lstm, test_x, test_y)
    device.authenticate_device(pred_labels)
    return test_loss, test_acc

# 云端进程
def cloud_process(queue):
    # 初始化 LSTM 模型
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"云端：使用设备 {device}")
    lstm = lstm.to(device)

    # 生成和保存训练数据
    train_x, train_y, device_train_list = generate_car_data(num_samples=NUM_SAMPLES_TRAIN, input_dim=INPUT_DIM, per_positive=PER_POSITIVE)
    train_x, train_y = train_x.to(device), train_y.to(device)
    save_data(train_x, train_y, device_train_list, prefix='train')

    # 模型训练
    lstm, train_loss_pos_list, train_accuracy_list, train_loss_list, train_epoch_list = run.train_model(
        lstm, train_x, train_y, max_epochs=TRAIN_EPOCHS)
    run.save_model(lstm)
    print("云端：模型训练完成")

    # 发送模型到边端
    cloud_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cloud_socket.bind(('localhost', 8000))
    cloud_socket.listen(1)
    print("云端：等待边端连接...")
    conn, addr = cloud_socket.accept()
    print(f"云端：边端连接 {addr}")
    
    send_data(conn, lstm)  # 发送模型数据
    conn.close()
    print("云端：模型发送给边端")

    # 接收微调后的测试集结果
    cloud_socket.listen(1)
    conn, addr = cloud_socket.accept()
    print(f"云端：收到边端发送的测试数据 {addr}")
    test_x, test_y = recv_data(conn)
    test_x, test_y = test_x.to(device), test_y.to(device)

    # 微调模型
    lstm, fine_tune_loss_list, fine_tune_accuracy_list, _, fine_tune_epoch_list = run.train_model(
        lstm, test_x, test_y, SFT_EPOCHS, lr=1e-4, weight_decay=1e-7)
    run.save_model(lstm)
    print("云端：微调完成")

    # 继续测试
    test_loss_cloud, test_acc_cloud, _, _, _=  run.test_model(lstm, test_x, test_y)
    print(f"test_acc_cloud={test_acc_cloud:.2f}")
    print(f"test_loss_cloud={test_loss_cloud:.5f}")
    conn.close()



# 边端进程
def edge_process(queue):
    print("边端：准备连接到云端")
    edge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    connected = False
    while not connected:
        try:
            edge_socket.connect(('localhost', 8000))  # 连接到云端
            print("边端：成功连接到云端")
            connected = True
        except ConnectionRefusedError:
            print("边端：无法连接到云端，重试中...")
            time.sleep(2)

    lstm = recv_data(edge_socket)  # 接收模型数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"边端：使用设备 {device}")
    lstm = lstm.to(device)
    edge_socket.close()
    print("边端：模型已接收")

    # 开始监听端口等待小车连接
    edge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    edge_socket.bind(('localhost', 8001))  # 边端监听
    edge_socket.listen(1)
    print("边端：已经开始监听端口等待小车发送测试集...")

    # 等待小车发送测试集
    conn, addr = edge_socket.accept()
    print(f"边端：收到来自小车的测试集 {addr}")
    
    test_x, test_y = recv_data(conn)
    test_x, test_y = test_x.to(device), test_y.to(device)
    conn.close()

    # 使用测试集测试模型
    test_loss_edge, test_acc_edge, _, _, _ = run.test_model(lstm, test_x, test_y)
    print(f"test_acc_edge={test_acc_edge:.2f}")
    print(f"test_loss_edge={test_loss_edge:.5f}")
    print(f"边端：测试完成")

    # 将测试结果放入队列，传回主进程
    queue.put((test_loss_edge.detach().cpu(), test_acc_edge.detach().cpu()))

    # 将测试集发送回云端
    edge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    edge_socket.connect(('localhost', 8000))  # 连接到云端
    send_data(edge_socket, (test_x, test_y))  # 发送测试数据
    edge_socket.close()
    print("边端：测试数据已发送回云端")

# 小车进程
def car_process():
    # 生成测试数据
    test_x, test_y, device_test_list = generate_car_data(num_samples=NUM_SAMPLES_TEST, input_dim=INPUT_DIM, per_positive=PER_POSITIVE)
    
    device = torch.device('cpu')
    test_x, test_y = test_x.to(device), test_y.to(device)
    save_data(test_x, test_y, device_test_list, prefix='test')

    # 确保边端进程已经在监听
    time.sleep(3)

    connected = False
    car_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while not connected:
        try:
            print("小车：正在连接到边端...")
            car_socket.connect(('localhost', 8001))  # 连接到边端
            print("小车：成功连接到边端，发送测试集...")
            connected = True
        except ConnectionRefusedError:
            print("小车：无法连接到边端，重试中...")
            time.sleep(2)

    send_data(car_socket, (test_x, test_y))  # 发送测试集
    car_socket.close()
    print("小车：测试集发送完成")

# 主函数，创建三个进程
def main():
    # 创建队列用于接收云端和边端进程的结果
    queue = Queue()

    cloud = Process(target=cloud_process, args=(queue,))
    edge = Process(target=edge_process, args=(queue,))
    car = Process(target=car_process)
    
    # 启动进程
    cloud.start()
    time.sleep(5)  # 确保云端进程先启动并完成监听
    edge.start()
    time.sleep(3)  # 确保边端进程启动并准备好接收数据
    car.start()

    # 等待所有进程结束
    cloud.join()
    edge.join()
    car.join()


if __name__ == "__main__":
    main()
