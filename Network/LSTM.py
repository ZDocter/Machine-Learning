import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size
                            , num_layers=num_layers
                            , batch_first=True
                            )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        output, _ = self.LSTM(input)
        output = output[:, -1, :]  # 只保留每个样本最后一个序列数据
        output = self.fc(output)
        return output


if __name__ == "__main__":
    lstm = RNN(100, 200, 2, 10)
    print(lstm)
