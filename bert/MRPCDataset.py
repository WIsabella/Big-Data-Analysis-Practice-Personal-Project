import csv
from torch.utils.data import Dataset

class MRPCDataset(Dataset):
    def __init__(self, path="dataset/msr_paraphrase_train.txt"):
        self.data = []
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) < 5:
                    continue
                label = int(row[0])
                s1 = row[3]
                s2 = row[4]
                self.data.append((s1, s2, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s1, s2, label = self.data[idx]
        return (s1, s2), label
