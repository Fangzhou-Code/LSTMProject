from peft import PeftModel, PeftConfig
import torch,time,run


# 计算平均损失和准确率
def compute_avg_loss_acc(model, test_x_list, test_y_list):
    total_loss, total_acc = 0, 0
    for test_x, test_y in zip(test_x_list, test_y_list):
        loss, acc, _, _, _ = run.test_model(model, test_x, test_y)
        total_loss += loss
        total_acc += acc
    return total_loss / len(test_x_list), total_acc / len(test_x_list)

# 冻结基础模型的参数
for param in car_lstm.parameters():
    param.requires_grad = False

# 添加微调适配层
class ForkliftAdapter(torch.nn.Module):
    def __init__(self, base_model):
        super(ForkliftAdapter, self).__init__()
        self.base_model = base_model
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(base_model.output_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, base_model.output_dim)
        )

    def forward(self, x):
        base_output = self.base_model(x)
        adapter_output = self.adapter(base_output)
        return adapter_output

# 包装基础模型
forklift_finetune_model = ForkliftAdapter(car_lstm)

# 设置微调的优化器和损失函数
optimizer = torch.optim.Adam(forklift_finetune_model.adapter.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# 微调训练函数
def fine_tune_model(model, train_x, train_y, epochs, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

# 微调部分代码优化
start_time = time.time()
fine_tune_model(forklift_finetune_model, forklift_finetune_x, forklift_finetune_y, SFT_EPOCHS, optimizer, criterion)
end_time = time.time()

finetune_time_forklift = end_time - start_time

# 评估模型性能
forklift_finetune_model.eval()
test_loss_forklift3, test_acc_forklift3 = compute_avg_loss_acc(forklift_finetune_model, forklift_test_x_list, forklift_test_y_list)
