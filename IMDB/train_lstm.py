import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import dataset_vocab
from lstm_model import ImdbModel
import pickle
import torch.nn.functional as F
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
def collate_fn(batch):

    reviews, labels = zip(*batch)
    reviews = torch.LongTensor([vocab.transform(i, max_len=500) for i in reviews])
    labels = torch.LongTensor(labels)
    return reviews, labels

def get_dataloader(train):
    imdb_dataset = dataset_vocab.ImdbDataset(train, sequence_max_len=100)
    return DataLoader(imdb_dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)


vocab = pickle.load(open("./models/vocab.pkl", "rb"))
num_embeddings = len(vocab)
padding_idx = vocab.PAD
log_dir = "logs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir)
imdb_model = ImdbModel(num_embeddings=num_embeddings, padding_idx=padding_idx).to(device)
imdb_dataset = dataset_vocab.ImdbDataset(True)

device = torch.device('cuda')
train_dataloader = get_dataloader(True)
test_dataloader = get_dataloader(False)
optimizer = Adam(imdb_model.parameters())
epoch = 100
best_acc = 0.0
best_epoch = 0
for i in range(epoch):
    train_loss = 0
    imdb_model.train()
    bar = tqdm(train_dataloader, total=len(train_dataloader))
    for idx, (data, target) in enumerate(bar):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = imdb_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))
    train_loss /= len(train_dataloader.dataset)
    writer.add_scalar("train_loss", train_loss, i+1)
    # 测试
    test_loss = 0
    correct = 0
    imdb_model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device)
            target = target.to(device)
            output = imdb_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, \n'
          'Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss,
        correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    writer.add_scalar("test_loss", test_loss, i+1)
    writer.add_scalar("Accuracy", 100.*correct, i+1)
    if correct > best_acc:
        best_acc = correct
        best_epoch = i + 1
        torch.save(imdb_model.state_dict(), "lstm_model.pkl")
        print(f"新的最佳模型已保存，准确率: {best_acc:.4f}")
    writer.flush()
writer.close()