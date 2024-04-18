import torch
from run import load_model


lstm = load_model()

test_x = torch.load('Dataset/traindataset.pt')
test_y = torch.load('Dataset/testlabels.pt')


test_y_pred = test_x[:, -1, :]  # 预测任务标签
test_y_clas = test_y  # 分类任务标签
test_x = test_x[:, :-1, :]

pred_y_pred, pred_y_clas = lstm(test_x)

print(pred_y_pred[:10, :])
print(test_y_pred[:10, :])
print(pred_y_clas[:10, :])
print(test_y_clas[:10, :])
