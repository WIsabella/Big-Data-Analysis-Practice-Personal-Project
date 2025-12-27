import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class MRPCStructuredDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.df = pd.read_csv(
            file_path,
            sep='\t',
            header=0,  #第0行为表头
            names=['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String'],  # 表头字段名
            on_bad_lines='skip',
        )
        # 过滤空值行和无效标签行（确保Quality是0或1）
        self.df = self.df.dropna(subset=['#1 String', '#2 String', 'Quality'])
        self.df = self.df[self.df['Quality'].isin([0, 1])]  # 只保留标签为0或1的行
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentence1 = str(row['#1 String'])
        sentence2 = str(row['#2 String'])
        label = int(row['Quality'])

        encoding = self.tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


# 加载分词器和模型
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 加载本地数据集
train_path = f"dataset/msr_paraphrase_train.txt"
dev_path = f"dataset/msr_paraphrase_test.txt"

try:
    train_dataset = MRPCStructuredDataset(train_path, tokenizer)
    dev_dataset = MRPCStructuredDataset(dev_path, tokenizer)
    print(f"训练集加载成功，共 {len(train_dataset)} 条数据")
    print(f"验证集加载成功，共 {len(dev_dataset)} 条数据")
except Exception as e:
    print(f"数据集加载失败：{e}")
    exit()


# 定义评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


# 训练参数
training_args = TrainingArguments(
    output_dir="./mrpc_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    no_cuda=False  # 若没有GPU，设为True
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()


# 推理函数
def predict(sentence1, sentence2):
    inputs = tokenizer(
        sentence1,
        sentence2,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.cuda()

    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=1).item()
    return "一致" if pred_label == 1 else "不一致"


# 测试示例
print("\n测试推理：")
print(predict("A man is playing guitar.", "Someone is playing a guitar."))
print(predict("The cat sits on the mat.", "A dog chases a ball."))