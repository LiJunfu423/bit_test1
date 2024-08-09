import torch
import torch.nn as nn
import torch.nn.functional as F

class GruModel(nn.Module):
    def __init__(self, num_embeddings, padding_idx):
        super(GruModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=200, padding_idx=padding_idx).to()
        self.gru = nn.GRU(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.5)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]
        output, h_n = self.gru(input_embeded)  # h_n :[4,batch_size,hidden_size]
        # out :[batch_size,hidden_size*2]
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出
        out_fc1 = self.fc1(out)
        out_fc1_relu = F.relu(out_fc1)

        out_fc2 = self.fc2(out_fc1_relu)  # out :[batch_size,2]
        return F.log_softmax(out_fc2, dim=-1)
