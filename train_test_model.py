import run
import time
import torch



if __name__ == "__main__":
    # 计算时间
    start_time = time.time() 

    # 初始化
    input_dim = 10
    per_positive = 0.6
    NEW_DATA = 0
    train_epochs = 500

    lstm, train_x, train_y, device_train_list = run.initialize_model_data(4, input_dim, per_positive, NEW_DATA)
    test_x, test_y, device_test_list = run.initialize_test_data(input_dim, per_positive, NEW_DATA) 


    # 加速（CPU记得注释掉）
    lstm = lstm.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda() 
    test_y = test_y.cuda()
        
    # 训练
    lstm, train_loss_pos_list, train_accuracy_list, train_loss_list, train_epoch_list = run.train_model(lstm, train_x, 
                                                                train_y, max_epochs=train_epochs)
    run.save_model(lstm)
    print("...Training Finished...")    
    end_time = time.time()
    execution_time = end_time - start_time


    # 测试
    lstm = run.load_model()
    device_id = "device123" 
    manufacturer = "Example Inc."
    device = run.DeviceAuthentication(device_id, manufacturer)# 实例化设备验证类
    

    test_loss, test_acc, pred_labels,  pred_y_clas, pred_labels = run.test_model(lstm, test_x, test_y)
    print("...Test Finished...") 
    print("test_acc=",test_acc)
    print("test_loss=",test_loss)
    device.authenticate_device(pred_labels)

    # 输出
    print(f"模型训练时间: {execution_time}秒")
    print('Test Loss: {:.5f}'.format(test_loss))
    print('Test Accuracy: {:.2f}%'.format(test_acc * 100)) 
    print("...Test Finished...")


    # 画图
    run.plot_curve(train_loss_pos_list, train_accuracy_list, train_loss_list, train_epoch_list) # 训练集
