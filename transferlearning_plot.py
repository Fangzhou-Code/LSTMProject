import matplotlib.pyplot as plt
import numpy as np

# 图一
labels = ['Forklift', 'Drone']
direct_accuracy = [0.3799999952316284, 0.3812398546378122]
transfer_accuracy = [0.4699999988079071, 0.4912358452211466]
# 使用自然风格的配色
nature_colors = ['#88CC88', '#44AA99']
# 绘制 AGV 模型直接使用与迁移学习后的准确率对比
plt.figure(figsize=(10, 6))
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, direct_accuracy, width, label='Before Transfer Learning', color=nature_colors[0])
plt.bar(x + width/2, transfer_accuracy, width, label='After Transfer Learning', color=nature_colors[1])
plt.ylabel('Accuracy')
plt.title('Before Transfer Learning vs After Transfer Learning')
plt.xticks(x, labels)
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
plt.show()


# 图二
finetuning_time = [3.1070029735565186, 2.87453952741125]
training_time = [43.42169713973999, 41.68123975423681]
# 使用自然风格的配色
nature_colors = ['#88CC88', '#44AA99']
# 绘制迁移学习 vs 重新训练时间对比
plt.figure(figsize=(10, 6))
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, finetuning_time, width, label='Transfer Learning Time', color=nature_colors[0])
plt.bar(x + width/2, training_time, width, label='Training Time', color=nature_colors[1])
plt.ylabel('Time (s)')
plt.title('Training Time vs Transfer Learning Time')
plt.xticks(x, labels)
plt.legend( ncol=2)
plt.show()

# 图三
labels = ['Forklift', 'Drone']
finetuning_accuracy = [0.4699999988079071, 0.4912358452211466]
training_accuracy = [0.5999999642372131, 0.6199999642372131]

# 使用自然风格的配色
nature_colors = ['#88CC88', '#44AA99']

# 绘制迁移学习 vs 重新训练准确率对比
plt.figure(figsize=(10, 6))
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, finetuning_accuracy, width, label='Transfer Learning Accuracy', color=nature_colors[0])
plt.bar(x + width/2, training_accuracy, width, label='Training Accuracy', color=nature_colors[1])

plt.ylabel('Accuracy')
plt.title('Transfer Learning Accuracy vs Training Accuracy')
plt.xticks(x, labels)
plt.legend( ncol=2)

plt.show()

