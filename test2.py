import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, confusion_matrix, f1_score
import logging
import time

# ================ seed ====================
parser = argparse.ArgumentParser(description='seed')
parser.add_argument('--seed', type=int, default=42, help='seed')
args = parser.parse_args()
seed = args.seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# ==================== log ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Seed: {seed}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== data pre-processing ====================
# lncRNA features
lncrna_feature = pd.read_csv('lncrna.feature.txt', sep='\s+', header=None)
for i in range(1, int(lncrna_feature[0].max()) + 1):
    idx = lncrna_feature[lncrna_feature[0] == i].index
    lncrna_feature.loc[idx, 1] /= lncrna_feature.loc[idx, 1].max()
lncrna_feature[0] /= lncrna_feature[0].max()

# gene features and network
gene_feature = pd.read_csv('gene.feature.txt', sep='\s+', header=None)
gene_net = pd.read_csv('gene-net1.txt', sep='\s+', header=None).values

# normalize gene features
for i in range(1, int(gene_feature[0].max()) + 1):
    idx = gene_feature[gene_feature[0] == i].index
    gene_feature.loc[idx, 1] /= gene_feature.loc[idx, 1].max()
gene_feature[0] /= gene_feature[0].max()

# ==================== GCN ====================
gene_net += np.eye(gene_net.shape[0])
D = gene_net.sum(axis=0)
D_hat = np.diag(D ** -0.5)
gene_gcn = D_hat @ gene_net @ D_hat @ gene_feature.values

# ==================== positive samples ====================
all_lncrna = pd.read_csv('all.lncrna.txt', header=None)[0].tolist()
all_gene = pd.read_csv('allgene.txt', header=None)[0].tolist()
lncrna_gene_net = pd.read_csv('lncrna-gene.net.txt', sep='\s+', header=None).values

pos = pd.read_csv('lncrna_target.txt', sep='\s+', header=None)
posdata, title = [], []
for _, row in pos.iterrows():
    try:
        idx1 = all_lncrna.index(row[2])
        idx2 = all_gene.index(row[3])
        features = np.concatenate([
            lncrna_feature.iloc[idx1].values,
            [lncrna_gene_net[idx1, idx2]],
            gene_gcn[idx2],
            [lncrna_gene_net[idx1, idx2]]
        ])
        posdata.append(features)
        title.append([row[2], row[3]])
    except Exception as e:
        continue

posdata = np.array(posdata, dtype=np.float32)
title = np.array(title)

# ==================== negative samples ====================
np.random.seed(seed)
negdata, title_neg = [], []
for j in range(len(posdata)):
    lnsample1 = np.random.choice(len(all_lncrna))
    current_ln = all_lncrna[lnsample1]

    # get associated genes
    associated_genes = pos[pos[2] == current_ln][3].tolist()
    gene_indices = [all_gene.index(g) for g in associated_genes if g in all_gene]

    # get available genes
    available = list(set(range(len(all_gene))) - set(gene_indices))
    if not available:
        continue
    lnsample2 = np.random.choice(available)

    # construct negative sample features
    features = np.concatenate([
        lncrna_feature.iloc[lnsample1].values,
        [lncrna_gene_net[lnsample1, lnsample2]],
        gene_gcn[lnsample2],
        [lncrna_gene_net[lnsample1, lnsample2]]
    ])
    negdata.append(features)
    title_neg.append([current_ln, all_gene[lnsample2]])

negdata = np.array(negdata, dtype=np.float32)
title_neg = np.array(title_neg)

# ==================== combine positive and negative samples ====================
X = np.vstack([posdata, negdata])
y = np.concatenate([np.ones(len(posdata)), np.zeros(len(negdata))])
all_titles = np.vstack([title, title_neg])
assert len(X) == len(all_titles), "Lengths of X and titles do not match."

# ==================== data split ====================
split_file = 'data_split.npz'

# load saved train/val/test split if exists
if os.path.exists(split_file):
    data_split = np.load(split_file, allow_pickle=True)
    X_train = data_split['X_train']
    X_val = data_split['X_val']
    X_test = data_split['X_test']
    y_train = data_split['y_train']
    y_val = data_split['y_val']
    y_test = data_split['y_test']
    titles_train = data_split['titles_train']
    titles_val = data_split['titles_val']
    titles_test = data_split['titles_test']

    logger.info(f"Loaded saved train/val/test split.")
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
# split data and save for next round
else:
    # train/val/test 72%/18%/10%
    # train 72%
    X_train, X_tmp, y_train, y_tmp, titles_train, titles_tmp = train_test_split(
        X, y, all_titles, test_size=0.28, stratify=y, random_state=seed
    )
    # val/test 18%/10%
    X_val, X_test, y_val, y_test, titles_val, titles_test = train_test_split(
        X_tmp, y_tmp, titles_tmp, test_size=10 / 28, stratify=y_tmp, random_state=seed
    )

    # save
    np.savez(split_file,
             X_train=X_train, X_val=X_val, X_test=X_test,
             y_train=y_train, y_val=y_val, y_test=y_test,
             titles_train=titles_train, titles_val=titles_val, titles_test=titles_test)

    logger.info(f"Saved train/val/test split.")

# ==================== model ====================
class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 4)),
            nn.Tanh(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=(2, 2)),
            nn.Tanh(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=(1, 2)),
            nn.Tanh(),
            nn.MaxPool2d((1, 1)),
            nn.Dropout(0.25)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            dummy = self.conv_layers(dummy)
            self.fc_input = dummy.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input, 512),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# ==================== training ====================
def reshape_data(data):
    reshaped = data.reshape(-1, 2, lncrna_feature.shape[1] + 1, 1)
    return torch.from_numpy(reshaped.transpose(0, 3, 1, 2)).float().contiguous()


X_train_tensor = reshape_data(X_train).to(device)
X_val_tensor = reshape_data(X_val).to(device)
X_test_tensor = reshape_data(X_test).to(device)
train_dataset = TensorDataset(X_train_tensor, torch.LongTensor(y_train).to(device))
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, torch.LongTensor(y_val).to(device))
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

model = CNNModel(input_shape=(2, lncrna_feature.shape[1] + 1)).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
best_model_path = "best_model.pth"
num_epochs = 30
durations = []
memory_usages = []

for epoch in range(num_epochs):
    start_time = time.time()

    # training
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)  # 计算训练集平均损失
    end_time = time.time()
    durations.append(end_time - start_time)
    memory_usages.append(torch.cuda.memory_allocated() / 1024 ** 2)

    # validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)

    # save best model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)  # 保存模型权重
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}]: New best model saved with Val Loss: {val_loss:.4f}")

    # log
    if (epoch + 1) % 5 == 0:
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] - '
                    f'Train Loss: {train_loss:.4f} - '
                    f'Val Loss: {val_loss:.4f} - ')

logger.info(f'Average training time: {np.mean(durations):.2f} s/epoch')
logger.info(f"Maximun memory usage: {max(memory_usages):.2f} MiB")

# ==================== evaluation ====================
model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():
    outputs = model(X_test_tensor)
    probas = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

# save results
results = np.column_stack([titles_test, y_test, probas])
pd.DataFrame(results, columns=['lncRNA', 'Gene', 'Label', 'Probability']).to_csv(f'predictions-{seed}.csv', index=False)

# metrics
roc_auc = roc_auc_score(y_test, probas)
precision, recall, _ = precision_recall_curve(y_test, probas)
pr_auc = auc(recall, precision)

tp, fp, tn, fn = confusion_matrix(y_test, probas > 0.5).ravel()
specificity = tn / (tn + fp)

f1 = f1_score(y_test, probas > 0.5)

logger.info(f'ROC AUC: {roc_auc:.4f}')
logger.info(f'PR AUC: {pr_auc:.4f}')
logger.info(f'Accuracy: {accuracy_score(y_test, probas > 0.5):.4f}')
logger.info(f'Specificity: {specificity:.4f}')
logger.info(f'F1 Score: {f1:.4f}')

# ==================== prediction ====================
# get test intersection pairs
biopathnet_test = pd.read_csv('biopathnet-test.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])
logger.info(f"BioPathNet positive test sample size: {biopathnet_test.shape[0]}")

positive_titles_test = titles_test[y_test == 1]
logger.info(f"Baseline positive test sample size: {len(positive_titles_test)}")

head = list(set(biopathnet_test['head']).intersection(positive_titles_test[:, 0]))
logger.info(f"Head intersection ({len(head)}): {head}")

tail = list(set(biopathnet_test['tail']).intersection(positive_titles_test[:, 1]))
logger.info(f"Tail intersection ({len(tail)}): {tail}")

# generate all possible pairs
all_pairs = [(h, t) for h in head for t in tail]

full_features = []
full_titles = []
full_labels = []
full_splits = []

for lnc, gene in all_pairs:
    try:
        lnc_idx = all_lncrna.index(lnc)
        gene_idx = all_gene.index(gene)
    except ValueError:
        continue

    feature = np.concatenate([
        lncrna_feature.iloc[lnc_idx].values,
        [lncrna_gene_net[lnc_idx, gene_idx]],
        gene_gcn[gene_idx],
        [lncrna_gene_net[lnc_idx, gene_idx]]
    ]).astype(np.float32)

    label = 1 if ((pos[2] == lnc) & (pos[3] == gene)).any() else 0
    current_title = np.array([lnc, gene])
    in_train = np.any((titles_train == current_title).all(axis=1))
    in_test = np.any((titles_test == current_title).all(axis=1))

    full_features.append(feature)
    full_titles.append([lnc, gene])
    full_labels.append(label)
    full_splits.append('train' if in_train else 'test' if in_test else 'unused')

    if len(full_features) % 1000 == 0:
        logger.info(f'Processed: {len(full_features)} / {len(all_pairs)}')

X_full = np.array(full_features)
X_full_tensor = reshape_data(X_full).to(device)

with torch.no_grad():
    outputs_full = model(X_full_tensor)
    probas_full = torch.softmax(outputs_full, dim=1)[:, 1].cpu().numpy()

full_results = pd.DataFrame({
    'lncRNA': [pair[0] for pair in full_titles],
    'Gene': [pair[1] for pair in full_titles],
    'Data_Split': full_splits,
    'Label': full_labels,
    'Probability': probas_full
})
full_results.to_csv(f'full-intersection-predictions-{seed}.csv', index=False)
