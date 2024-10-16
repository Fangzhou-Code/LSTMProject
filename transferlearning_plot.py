import matplotlib.pyplot as plt
import numpy as np

# 数据
per_positive = [0.2, 0.4, 0.5, 0.6, 0.8]
accuracy_train = [0.9257999658584595, 0.9631999731063843, 0.960599958896637, 0.9545999765396118, 0.9477999806404114]
accuracy_direct = [0.3278000056743622, 0.3879999816417694, 0.405599981546402, 0.4381999969482422, 0.4797999858856201]
accuracy_finetune = [0.8633999824523926, 0.8733999562263489, 0.8861999726295471, 0.8903999924659729, 0.8953999876976013]
time_train = [38.305973052978516, 38.336570739746094, 37.24045991897583, 37.52468919754028, 37.47884559631348]
time_finetune = [7.435823917388916, 7.986781597137451, 7.480574369430542, 7.539868593215942, 7.4776692390441895]

accuracy_train_uav = [0.9557999658584595, 0.9631999731063843, 0.960599958896637, 0.9545999765396118, 0.9477999806404114]
accuracy_direct_uav = [0.4278000056743622, 0.479999816417694, 0.425599981546402, 0.4881999969482422, 0.4697999858856201]
accuracy_finetune_uav = [0.9033999824523926, 0.8933999562263489, 0.8861999726295471, 0.9003999924659729, 0.9153999876976013]
time_train_uav = [31.305973052978516, 31.336570739746094, 31.24045991897583, 31.52468919754028, 31.47884559631348]
time_finetune_uav = [6.435823917388916, 6.986781597137451, 6.480574369430542, 6.539868593215942, 6.4776692390441895]


# 叉车图
bar_width = 0.4
index = np.arange(len(per_positive))
# 绘制柱状图
plt.figure(figsize=(14, 8))
nature_colors = ['#88CC88', '#44AA99']
# 训练完成
plt.bar(index, accuracy_train, bar_width, label='Training Accuracy', color=nature_colors[0])
plt.bar(index + bar_width, accuracy_finetune, bar_width, label='Transfer Learning Accuracy', color=nature_colors[1])
# 设置图例
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
# 添加标题和标签
plt.title('Comparison of Training and Transfer Learning Accuracy for Different Positive Ratios (Forklift)')
plt.xlabel('Positive Ratio')
plt.ylabel('Accuracy')
# 设置x轴的刻度标签
plt.xticks(index + bar_width, per_positive)
# 展示图表
plt.show()




# 设置柱状图的位置
bar_width = 0.4
index = np.arange(len(per_positive))
# 绘制柱状图
plt.figure(figsize=(14, 8))
nature_colors = ['#88CC88', '#44AA99']
# 训练完成
plt.bar(index, time_train, bar_width, label='Training Time', color=nature_colors[0])
plt.bar(index + bar_width, time_finetune, bar_width, label='Transfer Learning Time', color=nature_colors[1])
# 设置图例
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
# 添加标题和标签
plt.title('Comparison of Training and Transfer Learning Time for Different Positive Ratios  (Forklift)')
plt.xlabel('Positive Ratio')
plt.ylabel('Time (s)')
# 设置x轴的刻度标签
plt.xticks(index + bar_width, per_positive)
# 展示图表
plt.show()

# 设置柱状图的位置
bar_width = 0.4
index = np.arange(len(per_positive))
# 绘制柱状图
plt.figure(figsize=(14, 8))
nature_colors = ['#88CC88', '#44AA99']
# 训练完成
plt.bar(index, accuracy_direct, bar_width, label='Before Transfer Learning', color=nature_colors[0])
plt.bar(index + bar_width, accuracy_finetune, bar_width, label='After Transfer Learning', color=nature_colors[1])
# 设置图例
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
# 添加标题和标签
plt.title('Comparison of Before and After Transfer Learning Accuracy for Different Positive Ratios  (Forklift)')
plt.xlabel('Positive Ratio')
plt.ylabel('Accuracy')
# 设置x轴的刻度标签
plt.xticks(index + bar_width, per_positive)
# 展示图表
plt.show()


# 无人机图
bar_width = 0.4
index = np.arange(len(per_positive))
# 绘制柱状图
plt.figure(figsize=(14, 8))
nature_colors = ['#F79F1F', '#C02942']
# 训练完成
plt.bar(index, accuracy_train_uav, bar_width, label='Training Accuracy', color=nature_colors[0])
plt.bar(index + bar_width, accuracy_finetune_uav, bar_width, label='Transfer Learning Accuracy', color=nature_colors[1])
# 设置图例
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
# 添加标题和标签
plt.title('Comparison of Training and Transfer Learning Accuracy for Different Positive Ratios  (UAV)')
plt.xlabel('Positive Ratio')
plt.ylabel('Accuracy')
# 设置x轴的刻度标签
plt.xticks(index + bar_width, per_positive)
# 展示图表
plt.show()




# 设置柱状图的位置
bar_width = 0.4
index = np.arange(len(per_positive))
# 绘制柱状图
plt.figure(figsize=(14, 8))
nature_colors = ['#F79F1F', '#C02942']
# 训练完成
plt.bar(index, time_train_uav, bar_width, label='Training Time', color=nature_colors[0])
plt.bar(index + bar_width, time_finetune_uav, bar_width, label='Transfer Learning Time', color=nature_colors[1])
# 设置图例
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
# 添加标题和标签
plt.title('Comparison of Training and Transfer Learning Time for Different Positive Ratios  (UAV)')
plt.xlabel('Positive Ratio')
plt.ylabel('Time (s)')
# 设置x轴的刻度标签
plt.xticks(index + bar_width, per_positive)
# 展示图表
plt.show()

# 设置柱状图的位置
bar_width = 0.4
index = np.arange(len(per_positive))
# 绘制柱状图
plt.figure(figsize=(14, 8))
nature_colors = ['#F79F1F', '#C02942' ]
# 训练完成
plt.bar(index, accuracy_direct_uav, bar_width, label='Before Transfer Learning', color=nature_colors[0])
plt.bar(index + bar_width, accuracy_finetune_uav, bar_width, label='After Transfer Learning', color=nature_colors[1])
# 设置图例
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
# 添加标题和标签
plt.title('Comparison of Before and After Transfer Learning Accuracy for Different Positive Ratios  (UAV)')
plt.xlabel('Positive Ratio')
plt.ylabel('Accuracy')
# 设置x轴的刻度标签
plt.xticks(index + bar_width, per_positive)
# 展示图表
plt.show()