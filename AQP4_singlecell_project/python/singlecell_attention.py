import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention_layer = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        att = torch.tanh(self.attention_layer(x))
        score = self.context_vector(att)
        attention_weights = torch.softmax(score, dim=1)
        context = attention_weights * x
        return context, attention_weights


class NeuralNetWithAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, hidden_dim,
                 output_dim, treatment_classes):
        super().__init__()
        self.attention = Attention(input_dim, attention_dim)
        self.fc1 = nn.Linear(input_dim + treatment_classes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.treatment_classes = treatment_classes

    def forward(self, x, treatment):
        context, attention_weights = self.attention(x)
        treatment = treatment.squeeze()
        treatment_onehot = F.one_hot(
            treatment,
            num_classes=self.treatment_classes
        )
        combined = torch.cat((context, treatment_onehot.float()), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        return x, attention_weights


class CustomDataset(Dataset):
    def __init__(self, transcriptomes, treatments, targets):
        self.transcriptomes = transcriptomes
        self.treatments = treatments
        self.targets = targets

    def __len__(self):
        return len(self.transcriptomes)

    def __getitem__(self, idx):
        transcriptome = torch.tensor(
            self.transcriptomes[idx], dtype=torch.float32
        )
        treatment = torch.tensor(self.treatments[idx], dtype=torch.int64)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return transcriptome, treatment, target


def train_attention_model(model, data_loader, num_epochs=50, lr=1e-3):
    import torch.optim as optim
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        for x, treatment, target in data_loader:
            x = x.to(device)
            treatment = treatment.to(device)

            optimizer.zero_grad()
            outputs, _ = model(x, treatment)
            loss = criterion(outputs, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)

    return loss_history
