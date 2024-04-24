import matplotlib.pyplot as plt

# 错误率和测试准确率
error_rate = 0.3
test_acc = 80.0  # 假设测试准确率为80%

# 计算错误率和错误率与测试准确率的乘积
error_rate_times_test_acc = error_rate * (test_acc / 100)

# 柱状图数据
categories = ['Error Rate', 'Error Rate * Test Accuracy']
values = [0.3, error_rate_times_test_acc]

# 绘制柱状图
plt.bar(categories, values, color=['red', 'blue'])
plt.ylabel('Value')
plt.title('Error Rate vs. Error Rate * Test Accuracy')
plt.show()
