import torch

def calculate_accuracy(output, labels):
    # 获取每个样本概率最高的位数的索引
    _, predicted_indices = output.max(dim=1)
    
    # 将索引转换为对应的预测标签（one-hot 编码）
    predicted_labels = torch.zeros_like(output)
    predicted_labels.scatter_(1, predicted_indices.view(-1, 1), 1)
    
    # 计算准确率
    correct_predictions = (predicted_labels == labels)
    accuracy = correct_predictions.all(dim=1).float().mean() * 100
    return accuracy

# 示例输出和标签
output = torch.tensor([[0.8, 0.2, 0.3, 0.4],   # 示例输出
                       [0.2, 0.6, 0.4, 0.7],   
                       [0.4, 0.3, 0.9, 0.1]])
labels = torch.tensor([[1, 0, 0, 0],  # 示例标签
                       [0, 1, 0, 0],  
                       [0, 0, 1, 0]])

# 计算准确率
accuracy = calculate_accuracy(output, labels)
print("Accuracy:", accuracy.item(), "%")
