import time
import torch
import torch.optim as optim
import pickle
from Dataset import generate_car_data
from Model import LSTM
import run

def save_data(train_x, train_y, device_list, prefix='train'):
    torch.save(train_x, f'Dataset/{prefix}dataset.pt')
    torch.save(train_y, f'Dataset/{prefix}labels.pt')
    with open(f'Dataset/{prefix}devicelist.pkl', 'wb') as f:
        pickle.dump(device_list, f)
    print(f"...{prefix} dataset Create Finished...")

def main():
    # 计算时间
    start_time = time.time() 

    # 初始化
    input_dim = 10
    per_positive = 0.6
    train_epochs = 400
    sft_epochs = 100  # 设置 SFT 训练的轮次
    INPUT_SIZE = 3
    HIDDEN_SIZE = 64
    NUM_LAYERS = 3
    PRED_OUTPUT_SIZE = 3
    CLAS_OUTPUT_SIZE = 5

    # 初始化 LSTM 模型
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_OUTPUT_SIZE, CLAS_OUTPUT_SIZE)

    # 生成和保存训练数据
    train_x, train_y, device_train_list = generate_car_data(num_samples=1000, input_dim=input_dim, per_positive=per_positive)
    save_data(train_x, train_y, device_train_list, prefix='train')

    # 生成和保存测试数据
    test_x, test_y, device_test_list = generate_car_data(num_samples=100, input_dim=input_dim, per_positive=per_positive)
    save_data(test_x, test_y, device_test_list, prefix='test')

    # 加速（如果使用 GPU）
    if torch.cuda.is_available():
        lstm = lstm.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    # 训练模型
    lstm, train_loss_pos_list, train_accuracy_list, train_loss_list, train_epoch_list = run.train_model(
        lstm, train_x, train_y, max_epochs=train_epochs)
    run.save_model(lstm)
    print("...Training Finished...")    

    # 测试模型
    lstm = run.load_model()
    device_id = "device123" 
    manufacturer = "Example Inc."
    device = run.DeviceAuthentication(device_id, manufacturer)  # 实例化设备验证类
    test_loss_edge, test_acc_edge, pred_y_pred, pred_y_clas, pred_labels = run.test_model(lstm, test_x, test_y)
    device.authenticate_device(pred_labels)
    print("...Test Finished...") 
    print(f"test_acc={test_acc_edge:.2f}")
    print(f"test_loss={test_loss_edge:.5f}")

    # 微调模型，使用测试数据进行微调
    lstm, fine_tune_loss_list, fine_tune_accuracy_list, fine_tune_loss_list, fine_tune_epoch_list = run.train_model(lstm, test_x, test_y, sft_epochs, lr=1e-4, weight_decay=1e-7)
    run.save_model(lstm)
    print("...Fine-Tuning Finished...")    

    # 输出训练时间和测试结果
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"模型训练时间: {execution_time:.2f}秒")

    # 微调后的模型进行测试
    lstm = run.load_model()
    device_id = "device123" 
    manufacturer = "Example Inc."
    device = run.DeviceAuthentication(device_id, manufacturer)  # 实例化设备验证类
    test_loss_cloud, test_acc_cloud, pred_y_pred, pred_y_clas, pred_labels = run.test_model(lstm, test_x, test_y)
    device.authenticate_device(pred_labels)
    print("...Test Finished...") 
    print(f"test_acc={test_acc_cloud:.2f}")
    print(f"test_loss={test_loss_cloud:.5f}")

    # 画训练曲线
    run.plot_curve(train_loss_pos_list, train_accuracy_list, train_loss_list, train_epoch_list)
    run.plot_curve(fine_tune_loss_list, fine_tune_accuracy_list, fine_tune_loss_list, fine_tune_epoch_list)
    # 画测试对比柱状图
    run.plot_comparison(test_loss_edge, test_acc_edge, test_loss_cloud, test_acc_cloud)
if __name__ == "__main__":
    main()
