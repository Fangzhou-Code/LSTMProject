import run
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

start_time = time.time() 

# 初始化
input_dim = 20
lstm, train_x, train_y, device_train_list = run.initialize_model_data(3, input_dim, 0)
train_epochs = 500
per_positive = 0.6

# 加速（CPU记得注释掉）
lstm = lstm.cuda()
train_x = train_x.cuda()
train_y = train_y.cuda()
    
# 训练
train_loss_pos_list, train_accuracy_list, train_loss_list, train_epoch_list = run.train_model(lstm, train_x, 
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
test_x, test_y, device_test_list = run.initialize_test_data(input_dim, 0) 
test_x = test_x.cuda() 
test_y = test_y.cuda()
test_loss, test_acc, pred_labels = run.test_model(lstm, test_x, test_y)
print("...Test Finished...") 
print("test_acc=",test_acc)
print("test_loss=",test_loss)
device.authenticate_device(pred_labels)

# 输出
print(f"模型训练时间: {execution_time}秒")
print('Test Loss: {:.5f}'.format(test_loss))
print('Test Accuracy: {:.2f}%'.format(test_acc * 100)) 
print("...Test Finished...")

# copy tensor to host memory
test_acc = test_acc.detach().cpu()
test_loss = test_loss.detach().cpu()

# 画图
run.plot_curve(train_loss_pos_list, train_accuracy_list, train_loss_list, train_epoch_list) # 训练集


# 对比实验
'''
负样本：被控制和被冒充
* 被控制：静态改变，行为改变 50%
* 被冒充：静态不变，行为改变 50%

设备指纹作为对比实验，准确率的计算是正样本+被冒充
我们提出的方案，准确率的计算是测试集acc
'''

ft_acc = sum(1 for k in device_test_list if k == "none") / len(device_test_list) + per_positive
print("设备指纹准确率=", ft_acc)
values = [per_positive * 100, ft_acc * 100, test_acc  * 100]
# 柱状图数据
categories = ['正样本数', '设备指纹', 'LSTM更新身份凭证']
# 绘制柱状图
plt.bar(categories, values, color=['#FFA07A', '#87CEEB', '#4682B4'])
plt.ylabel('Percentage')  # 纵坐标标签改为百分比
# 设置中文字体
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title(u'设备身份验证的准确率')
plt.ylim(0, 100)
plt.gca().yaxis.set_major_formatter(PercentFormatter())  # 设置纵坐标格式为百分比
plt.show()
plt.savefig('./results/comparison_bar.png')
