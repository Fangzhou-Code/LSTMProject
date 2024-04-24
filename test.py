import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# 计算错误率和错误率与测试准确率的乘积
error_rate = 1 - 0.7
error_rate_times_test_acc = error_rate * (0.9185 / 100)

# 柱状图数据
categories = ['Original', 'LSTM']
# 将错误率和错误率与测试准确率的乘积转换为百分比
values = [0.7 * 100, (1 - error_rate_times_test_acc) * 100]

# 绘制柱状图
plt.bar(categories, values, color=['#FFA07A', '#87CEEB'])
plt.ylabel('Percentage')  # 纵坐标标签改为百分比
plt.title('Error sample detection rate')
plt.gca().yaxis.set_major_formatter(PercentFormatter())  # 设置纵坐标格式为百分比
plt.ylim(0, 100)
plt.show()
