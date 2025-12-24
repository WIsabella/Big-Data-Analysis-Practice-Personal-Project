import torch
from torch.utils.data import DataLoader
from FCModel import FCModel
from MRPCDataset import MRPCDataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

# ============================
# 数据
# ============================
train_dataset = MRPCDataset()
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
print("数据载入完成")

# ============================
# 配置设备
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 加载预训练 BERT
# ============================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
print("BERT 模型加载完成")

# ============================
# 创建分类器 (全连接层)
model = FCModel().to(device)
print("全连接层创建完成")

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)      # 用于 FC 层
bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=2e-5)  # 用于 BERT

# Scheduler
EPOCHS = 3
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    bert_optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# 损失函数 & 准确率
crit = torch.nn.BCEWithLogitsLoss()

def binary_accuracy(pred, y):
    preds = torch.round(pred)
    correct = (preds == y).float()
    return correct.sum() / len(correct)

# 训练函数
def train():
    bert_model.train()
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    total = 0

    for i, (sentence, label) in enumerate(train_loader):
        label = label.to(device).float()

        # Tokenizer 编码句子对
        sentence1, sentence2 = sentence
        encoding = tokenizer(
            sentence1,
            sentence2,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)

        # BERT 前向传播
        bert_output = bert_model(**encoding)
        pooled = bert_output.pooler_output  # [batch, 768]

        # 分类
        predict = model(pooled).squeeze(1)  # [batch]

        # loss
        loss = crit(predict, label)

        # 概率 & 准确率
        prob = torch.sigmoid(predict)
        acc = binary_accuracy(prob, label)

        # 反向传播 + 更新
        optimizer.zero_grad()
        bert_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bert_optimizer.step()
        scheduler.step()

        epoch_loss += loss.item() * len(label)
        epoch_acc += acc.item() * len(label)
        total += len(label)

        print(f"batch {i} loss={loss.item():.4f} acc={acc.item():.4f}")

    return epoch_loss / total, epoch_acc / total

# 训练入口
for epoch in range(EPOCHS):
    loss, acc = train()
    print(f"⭐ EPOCH {epoch + 1} END — loss: {loss:.4f}  acc: {acc:.4f}")

print("训练完成！")
