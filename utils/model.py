from torch import nn
import torch


class NLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NLP, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.Softmax()

    def forward(self, x):
        # inicializando memória: L x B x F
        h = torch.zeros(1, 1, self.hidden_size)

        x = x.unsqueeze(1)

        output, h = self.gru(x, h)

        output = self.logsoftmax(self.linear(output[-1]))
        return output