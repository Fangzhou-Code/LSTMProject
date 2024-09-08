test_acc_edge=0.70
test_loss_edge=36.50026
test_acc_cloud=0.80
test_loss_cloud=49.48302

import matplotlib.pyplot as plt
import numpy as np

labels = ['Edge', 'Cloud']
losses = [test_loss_edge, test_loss_cloud]
accuracies = [test_acc_edge, test_acc_cloud]

x = np.arange(len(labels))  # 横轴刻度的位置
width = 0.35  # 柱状图的宽度

# 创建图形和子图
fig, ax1 = plt.subplots(figsize=(8, 6))

# 绘制损失的柱状图
ax1.bar(x - width/2, losses, width, label='Loss', color='skyblue')
ax1.set_xlabel('Model Deployment')
ax1.set_ylabel('Loss', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.legend(loc='upper left')

# 在同一个子图中共享 x 轴绘制准确率的柱状图
ax2 = ax1.twinx()  # 使用相同的 x 轴
ax2.bar(x + width/2, accuracies, width, label='Accuracy', color='orange')
ax2.set_ylabel('Accuracy', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.legend(loc='upper right')

# 添加标题和标签
plt.title('Comparison of Test Results between Edge and Cloud Deployment')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# 自动调整布局
fig.tight_layout()

# 展示图像
plt.show()