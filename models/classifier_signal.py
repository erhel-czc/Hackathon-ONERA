import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class IQDetectionDataset(Dataset):
    def __init__(self, n_samples: int, n_points: int,
                 snr_db_range: tuple[float, float] = (-10.0, 10.0),
                 seed: int | None = None):
        self.n_points = n_points
        rng = np.random.default_rng(seed)

        self.waveforms = np.zeros((n_samples, 2 * n_points), dtype=np.float32)
        self.labels = np.zeros(n_samples, dtype=np.float32)
        self.snr_db = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            label = rng.integers(0, 2)
            noise = (rng.standard_normal(n_points) + 1j * rng.standard_normal(n_points)) / np.sqrt(2)

            if label == 1:
                snr_db = rng.uniform(*snr_db_range)
                snr_lin = 10 ** (snr_db / 10)
                f0 = rng.uniform(0.05, 0.45)
                phase0 = rng.uniform(0, 2 * np.pi)
                t = np.arange(n_points)
                signal = np.exp(1j * (2 * np.pi * f0 * t + phase0))
                signal *= np.sqrt(snr_lin)
                x = signal + noise
                self.snr_db[i] = snr_db
            else:
                x = noise
                self.snr_db[i] = -np.inf

            x = x / (np.linalg.norm(x) + 1e-8)
            self.waveforms[i, :n_points] = x.real
            self.waveforms[i, n_points:] = x.imag
            self.labels[i] = label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.waveforms[idx]),
                torch.tensor(self.labels[idx]),
                torch.tensor(self.snr_db[idx]))


class SignalDetectorIQ(nn.Module):
    def __init__(self, window_size: int, in_channels: int = 2):
        super().__init__()
        self.window_size = window_size
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.view(b, 2, self.window_size)
        x = self.features(x)
        return self.classifier(x.squeeze(-1)).squeeze(-1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_logits, all_labels, all_snr = [], [], []
    for x, y, snr in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += ((logits > 0) == (y > 0.5)).sum().item()
        total += x.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
        all_snr.append(snr)
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "logits": torch.cat(all_logits),
        "labels": torch.cat(all_labels),
        "snr_db": torch.cat(all_snr),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WINDOW_SIZE = 4092
    N_TOTAL = 20000
    BATCH_SIZE = 128
    EPOCHS = 30
    LR = 1e-3

    dataset = IQDetectionDataset(N_TOTAL, WINDOW_SIZE, snr_db_range=(-15, 10), seed=42)
    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    n_test = len(dataset) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = SignalDetectorIQ(window_size=WINDOW_SIZE, in_channels=2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"epoch {epoch:03d}  train_loss {train_loss:.4f}  "
              f"val_loss {val_metrics['loss']:.4f}  val_acc {val_metrics['acc']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), "best_model.pt")

    model.load_state_dict(torch.load("best_model.pt"))
    test_metrics = evaluate(model, test_loader, criterion, device)

    y_true = test_metrics["labels"].numpy()
    y_score = torch.sigmoid(test_metrics["logits"]).numpy()
    auc = roc_auc_score(y_true, y_score)
    print(f"\ntest_acc {test_metrics['acc']:.4f}  test_AUC {auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    target_pfa = 0.01
    idx = np.searchsorted(fpr, target_pfa)
    threshold_at_pfa = thresholds[idx] if idx < len(thresholds) else 1.0

    snr_values = test_metrics["snr_db"].numpy()
    mask_h1 = y_true == 1
    snr_bins = np.arange(-15, 12, 3)
    print(f"\nPd par bin de SNR (seuil calibré à Pfa={target_pfa}):")
    for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
        bin_mask = mask_h1 & (snr_values >= lo) & (snr_values < hi)
        if bin_mask.sum() > 0:
            pd = (y_score[bin_mask] >= threshold_at_pfa).mean()
            print(f"  SNR [{lo:+.0f}, {hi:+.0f}) dB : Pd = {pd:.3f}  (n={bin_mask.sum()})")

    model.eval()
    example_input = torch.zeros(1, 2 * WINDOW_SIZE)
    scripted = torch.jit.trace(model, example_input)
    scripted.save("signal_detector_iq.pt")
    print("\nModèle exporté : signal_detector_iq.pt")


if __name__ == "__main__":
    main()