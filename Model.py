import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_output_size, clas_output_size):
        super(LSTM, self).__init__()
        # Parameters
        self.input_size = input_size  # Feature size
        self.hidden_size = hidden_size  # Number of hidden units
        self.num_layers = num_layers  # Number of LSTM layers to stack
        self.pred_output_size = pred_output_size  # Number of output for prediction task
        self.clas_output_size = clas_output_size  # Number of output for classification task
        self.bias = True
        self.batch_first = True
        self.dropout = 0.2 if self.num_layers > 1 else 0
        self.bidirectional = False
        # LSTM Layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, self.bias, self.batch_first, self.dropout, self.bidirectional)
        # Fully Connected Layers
        self.fc = nn.Linear(self.hidden_size, 128)
        self.fc1_pred = nn.Linear(128, 64)  # 预测下一个时间步的行为 position
        self.fc2_pred = nn.Linear(64, self.pred_output_size)  # 预测下一个时间步的行为 position
        self.fc1_clas = nn.Linear(128, 64)  # 对设备行为模式的分类 behavior
        self.fc2_clas = nn.Linear(64, self.clas_output_size)  # 对设备行为模式的分类 behavior
        # Loss Function
        self.loss_mse = nn.MSELoss()  # MSE用于预测下一步
        self.loss_ce = nn.CrossEntropyLoss()  # CE用于分类
        # Initial trainable hidden unit h0 and memory unit c0
        self.h0 = nn.Parameter(torch.zeros(self.num_layers, 10000, self.hidden_size))
        self.c0 = nn.Parameter(torch.zeros(self.num_layers, 10000, self.hidden_size))

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device=x.device)
        # c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device=x.device)
        # out, _ = self.lstm(x, (self.h0, self.c0))

        batch_size = x.size(0)
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_time_step = lstm_out[:, -1, :]

        # 预测任务
        fc_pred_out = self.fc1_pred(last_time_step)
        pred = self.fc2_pred(fc_pred_out)

        # 分类任务
        fc_clas_out = self.fc1_clas(last_time_step)
        clas = nn.Softmax(self.fc2_clas(fc_clas_out))

        return pred, clas
