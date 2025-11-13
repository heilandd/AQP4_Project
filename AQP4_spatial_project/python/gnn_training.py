from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from .gnn_models import EarlyStopping  # If you want EarlyStopping here, move it.


class EarlyStopping:
    \"\"\"Early stopping on validation loss.\"\"\"

    def __init__(self, patience=5, delta=0.0, checkpoint_path="checkpoint.pt"):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            torch.save(model.state_dict(), self.checkpoint_path)


def train_regressor(
    model,
    train_loader,
    val_loader,
    *,
    criterion1,
    criterion2,
    optimizer,
    num_epochs: int = 50,
    patience: int = 5,
    checkpoint_path: str = "checkpoint.pt",
    device=None,
):
    \"\"\"Train model on regression + classification tasks.\"\"\"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    es = EarlyStopping(patience=patience, delta=0.01, checkpoint_path=checkpoint_path)

    loss_train = []
    loss_val = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for data in tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False):
            data = data.to(device)
            optimizer.zero_grad()

            data.edge_attr = data.distance.view(-1, 1).float()
            _, cls_out, pred_reg, _, _ = model(data)

            target_reg = data.region.float()
            loss_reg = criterion1(pred_reg, target_reg)

            target_cls = data.status.long()
            loss_cls = criterion2(cls_out, target_cls)

            loss = 0.5 * (loss_reg + loss_cls)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * data.num_graphs

        train_loss = epoch_loss / len(train_loader.dataset)
        loss_train.append(train_loss)

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.edge_attr = data.distance.view(-1, 1).float()
                _, _, pred_reg, _, _ = model(data)
                val_preds.append(pred_reg.cpu().numpy())
                val_true.append(data.region.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_loss = mean_squared_error(val_true, val_preds)
        loss_val.append(val_loss)

        print(f"[{epoch:02d}] train MSE: {train_loss:.4f}   val MSE: {val_loss:.4f}")

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, loss_train, loss_val
