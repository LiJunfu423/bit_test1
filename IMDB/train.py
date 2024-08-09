# import torch
# import torch.nn as nn
# from torch import optim
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchtext.data import get_tokenizer
# from torchtext.datasets import IMDB
# import random
# import numpy as np
# from tqdm import tqdm
# import lstm
# from collections import Counter, OrderedDict
# from torchtext.vocab import vocab


# def binary_accuracy(preds, y):
#     # 将预测结果使用 sigmoid 激活函数
#     rounded_preds = torch.round(torch.sigmoid(preds))
#     # 比较预测结果和真实标签
#     correct = (rounded_preds == y).float()
#     # 计算准确率
#     acc = correct.sum() / len(correct)
#     return acc
#
#
# def yield_tokens(data_iter):
#     for _, text in data_iter:
#         yield tokenizer(text)
#
#
# def collate_batch(batch):
#     label_list, text_list = [], []
#     for (_label, _text) in batch:
#         label_list.append(label_transform(_label))
#         processed_text = torch.tensor(text_transform(_text))
#         text_list.append(processed_text)
#     return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)
#
#
# def batch_sampler(batch_size):
#     indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(train_list)]
#     random.shuffle(indices)
#     pooled_indices = []
#     # create pool of indices with similar lengths
#     for i in range(0, len(indices), batch_size * 100):
#         pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
#
#     pooled_indices = [x[0] for x in pooled_indices]
#
#     # yield indices for current batch
#     for i in range(0, len(pooled_indices), batch_size):
#         yield pooled_indices[i:i + batch_size]
#
#
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda")
# torch.backends.cudnn.deterministic = True
# # SEED = 1234
# # random.seed(SEED)
# # np.random.seed(SEED)
# # torch.cuda.manual_seed(SEED)
#
# train_data = IMDB(split='train')
# test_data = IMDB(split='test')
# tokenizer = get_tokenizer('basic_english')
# train_iter = iter(train_data)
# test_iter = iter(test_data)
#
# counter = Counter()
# for (label, line) in train_iter:
#     counter.update(tokenizer(line))
# sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
# ordered_dict = OrderedDict(sorted_by_freq_tuples)
# vocab = vocab(ordered_dict, min_freq=10, specials=['<unk>', '<BOS>', '<EOS>', '<PAD>'])
# vocab.set_default_index(-1)
# vocab.set_default_index(vocab['<unk>'])
#
# text_transform = lambda x: [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']]
# label_transform = lambda x: 1 if x == 'pos' else 0
#
#
# train_dataloader = DataLoader(list(train_iter), batch_size=64, shuffle=True,
#                               collate_fn=collate_batch)
#
# test_dataloader = DataLoader(list(test_iter), batch_size=64, shuffle=True,
#                              collate_fn=collate_batch)
#
#
#
# # 参数
# input_size = len(vocab)
# hidden_size = 256
# output_size = 1
# num_layers = 4
# BATCH_SIZE = 64
# learning_rate = 0.01
# model = lstm.LSTMModel(input_size, hidden_size, output_size, num_layers).to('cuda')  # 在GPU上训练
# criterion = nn.SmoothL1Loss()
# log_dir = "logs"
# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
# epoch = 100
# writer = SummaryWriter(log_dir)
# best_acc = 0.0
# best_epoch = 0
#
# for i in range(epoch):
#     print("第{}轮训练开始".format(i + 1))
#     model.train()
#     epoch_loss = 0
#     epoch_acc = 0
#     train_loader_tqdm = tqdm(train_dataloader, desc=f"训练轮次 {i + 1}/{epoch}", unit="batch")
#
#     for batch in train_dataloader:
#         texts, labels = batch
#         texts, labels = texts.to(device), labels.to(device)
#         optimizer.zero_grad()
#         predictions = model(texts).squeeze(1)
#         loss = criterion(predictions, labels)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         epoch_acc += binary_accuracy(predictions, labels).item()
#
#     train_loss = epoch_loss / len(train_dataloader)
#     train_acc = epoch_acc / len(train_dataloader)
#
#     writer.add_scalar("train_loss", train_loss, i)
#     writer.add_scalar("train_acc", train_acc, i)
#
#     epoch_loss = 0
#     epoch_acc = 0
#     model.eval()
#     test_loader_tqdm = tqdm(test_dataloader, desc=f"测试轮次 {i + 1}/{epoch}", unit="batch")
#     with torch.no_grad():
#         for batch in test_dataloader:
#             texts, labels = batch
#             texts, labels = texts.to(device), labels.to(device)
#             predictions = model(texts).squeeze(1)
#             loss = criterion(predictions, labels)
#             epoch_loss += loss.item()
#             epoch_acc += binary_accuracy(predictions, labels).item()
#     valid_loss = epoch_loss / len(test_dataloader)
#     valid_acc = epoch_acc / len(test_dataloader)
#     writer.add_scalar("test_loss", valid_loss, i)
#     writer.add_scalar("test_acc", valid_acc, i)
#     print(
#         f'| Epoch: {i + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% |')
#     if valid_acc > best_acc:
#         best_acc = valid_acc
#         best_epoch = i + 1
#         torch.save(model.state_dict(), "lstm_model.pt")
#         print(f"新的最佳模型已保存，准确率: {best_acc:.4f}")
#
#     writer.flush()
# writer.close()
