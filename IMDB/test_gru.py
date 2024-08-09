import pickle

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import dataset_vocab
from gru_model import GruModel


def collate_fn(batch):
    reviews, labels = zip(*batch)
    reviews = torch.LongTensor([vocab.transform(i, max_len=500) for i in reviews])
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataloader(train):
    imdb_dataset = dataset_vocab.ImdbDataset(train, sequence_max_len=100)
    return DataLoader(imdb_dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)


test_dataloader = get_dataloader(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = pickle.load(open("./models/vocab.pkl", "rb"))
num_embeddings = len(vocab)
padding_idx = vocab.PAD
test_model = GruModel(num_embeddings=num_embeddings, padding_idx=padding_idx).to(device)#  torch.model("gru_model.pkl").to(device)
test_model.load_state_dict(torch.load("gru_model.pkl"))
imdb_dataset = dataset_vocab.ImdbDataset(True)
device = torch.device('cuda')
optimizer = Adam(test_model.parameters())
test_loss = 0
correct = 0
test_model.eval()
with torch.no_grad():
    for data, target in tqdm(test_dataloader):
        data = data.to(device)
        target = target.to(device)
        output = test_model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
        correct += pred.eq(target.data.view_as(pred)).sum()
test_loss /= len(test_dataloader.dataset)
print('\nTest set: Avg. loss: {:.4f}, \n'
      'Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss,
    correct, len(test_dataloader.dataset),
    100. * correct / len(test_dataloader.dataset)))
