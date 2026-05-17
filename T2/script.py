import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA = "./fish_image"
INPUT_SIZE = 224
BATCH_SIZE = 16
WORKERS = 4
RANDOM_SEED = 42
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PLOT_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def countParameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def stepLrScheduler(scheduler, validation_loss):
    if scheduler is None:
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_loss)
    else:
        scheduler.step()


def replaceFcHead(model, num_classes, dropout=0.4):
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return model


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class CNNArch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBnRelu(3, 32),
            ConvBnRelu(32, 32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            ConvBnRelu(32, 64),
            ConvBnRelu(64, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            ConvBnRelu(64, 128),
            ConvBnRelu(128, 128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            ConvBnRelu(128, 256),
            ConvBnRelu(256, 256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.spatial_pool(x)
        return self.head(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depth = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.point = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.point(self.depth(x))


class MobileNetArch(nn.Module):
    def __init__(self, num_classes, width_mult=1.0):
        super().__init__()

        def Scaled(ch):
            return int(ch * width_mult)

        s = Scaled
        self.backbone = nn.Sequential(
            nn.Conv2d(3, s(32), 3, 2, 1, bias=False),
            nn.BatchNorm2d(s(32)),
            nn.ReLU(inplace=True),
            SeparableConv2d(s(32), s(64), stride=1),
            SeparableConv2d(s(64), s(128), stride=2),
            SeparableConv2d(s(128), s(128), stride=1),
            SeparableConv2d(s(128), s(256), stride=2),
            SeparableConv2d(s(256), s(256), stride=1),
            SeparableConv2d(s(256), s(512), stride=2),
            SeparableConv2d(s(512), s(512), stride=1),
            SeparableConv2d(s(512), s(512), stride=1),
            SeparableConv2d(s(512), s(512), stride=1),
            SeparableConv2d(s(512), s(512), stride=1),
            SeparableConv2d(s(512), s(512), stride=1),
            SeparableConv2d(s(512), s(1024), stride=2),
            SeparableConv2d(s(1024), s(1024), stride=1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(s(1024), num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        return self.head(x)


def makeResNetModel(num_classes, unfreeze_blocks=2):
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in net.parameters():
        param.requires_grad = False

    if unfreeze_blocks > 0:
        layer_stack = list(net.children())
        for block in layer_stack[-(unfreeze_blocks + 1) : -1]:
            for param in block.parameters():
                param.requires_grad = True

    replaceFcHead(net, num_classes)
    trainable = countParameters(net, trainable_only=True)
    total = countParameters(net)
    print(f"ResNetModel | Trainable: {trainable:,} / Total: {total:,} params")
    return net


def makeGoogLeNetModel(num_classes, unfreeze_blocks=1):
    net = models.googlenet(
        weights=models.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=False
    )

    for param in net.parameters():
        param.requires_grad = False

    if unfreeze_blocks >= 1:
        for param in net.inception5b.parameters():
            param.requires_grad = True
    if unfreeze_blocks >= 2:
        for param in net.inception5a.parameters():
            param.requires_grad = True

    replaceFcHead(net, num_classes)
    trainable = countParameters(net, trainable_only=True)
    total = countParameters(net)
    print(f"GoogLeNetModel | Trainable: {trainable:,} / Total: {total:,} params")
    return net


def buildAugmentationPipelines():
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_pipeline = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_pipeline = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_pipeline, eval_pipeline


def buildDataLoaders(root=DATA):
    train_pipeline, eval_pipeline = buildAugmentationPipelines()

    train_source = datasets.ImageFolder(root=root, transform=train_pipeline)
    eval_source = datasets.ImageFolder(root=root, transform=eval_pipeline)

    labels = train_source.classes
    n_classes = len(labels)
    n_images = len(train_source)

    n_train = int(SPLIT_TRAIN * n_images)
    n_val = int(SPLIT_VAL * n_images)
    n_test = n_images - n_train - n_val

    rng = torch.Generator().manual_seed(RANDOM_SEED)
    shuffled_idx = torch.randperm(n_images, generator=rng).tolist()

    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train : n_train + n_val]
    test_idx = shuffled_idx[n_train + n_val :]

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True)
    train_loader = DataLoader(
        torch.utils.data.Subset(train_source, train_idx),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(eval_source, val_idx),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(eval_source, test_idx),
        shuffle=False,
        **loader_kwargs,
    )

    print()
    print(f"Dataset loaded: {n_images} images | {n_classes} classes")
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"Classes: {labels}")
    print()

    return train_loader, val_loader, test_loader, n_classes, labels


def trainEpoch(model, loader, loss_fn, optimizer):
    model.train()
    loss_sum, hits, n_samples = 0.0, 0, 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch_x.size(0)
        hits += logits.argmax(1).eq(batch_y).sum().item()
        n_samples += batch_y.size(0)

    return loss_sum / n_samples, hits / n_samples


def validateEpoch(model, loader, loss_fn):
    model.eval()
    loss_sum, hits, n_samples = 0.0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss_sum += loss.item() * batch_x.size(0)
            hits += logits.argmax(1).eq(batch_y).sum().item()
            n_samples += batch_y.size(0)

    return loss_sum / n_samples, hits / n_samples


def fitModel(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler=None,
    epochs=30,
    run_label="model",
):
    model = model.to(device)
    metrics_log = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    peak_val_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        tr_loss, tr_acc = trainEpoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_acc = validateEpoch(model, val_loader, loss_fn)

        metrics_log["train_loss"].append(tr_loss)
        metrics_log["val_loss"].append(va_loss)
        metrics_log["train_acc"].append(tr_acc)
        metrics_log["val_acc"].append(va_acc)

        stepLrScheduler(scheduler, va_loss)

        if va_acc > peak_val_acc:
            peak_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{run_label}] Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}"
        )

    elapsed = time.time() - t0
    metrics_log["training_time"] = elapsed
    print()
    print(
        f"{run_label} | Training time: {elapsed:.1f}s | "
        f"Best Val Acc: {peak_val_acc:.4f}"
    )
    print()

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{run_label}_best.pth")
    return model, metrics_log


def collectPredictions(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(batch_y.numpy())

    return np.array(targets), np.array(preds)


def evaluateOnTest(model, test_loader, labels, run_label, loss_fn):
    test_loss, test_acc = validateEpoch(model, test_loader, loss_fn)
    y_true, y_pred = collectPredictions(model, test_loader)

    print()
    print("=================================================")
    print(f"  {run_label} — Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("=================================================")
    print(classification_report(y_true, y_pred, target_names=labels))

    saveConfusionMatrix(y_true, y_pred, labels, run_label)
    savePerClassBarChart(y_true, y_pred, labels, run_label)

    return test_loss, test_acc, y_true, y_pred


def saveConvergencePlot(logs, run_labels, output_file="convergence.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (log, label) in enumerate(zip(logs, run_labels)):
        epoch_axis = range(1, len(log["train_loss"]) + 1)
        color = PLOT_PALETTE[idx]
        axes[0].plot(epoch_axis, log["train_loss"], "--", color=color, alpha=0.6)
        axes[0].plot(epoch_axis, log["val_loss"], "-", color=color, label=label)
        axes[1].plot(epoch_axis, log["train_acc"], "--", color=color, alpha=0.6)
        axes[1].plot(epoch_axis, log["val_acc"], "-", color=color, label=label)

    for ax, heading, ylab in zip(
        axes,
        ["Loss Convergence", "Accuracy Convergence"],
        ["Loss", "Accuracy"],
    ):
        ax.set_title(f"{heading} (solid=val, dashed=train)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def saveConfusionMatrix(y_true, y_pred, labels, run_label):
    matrix = confusion_matrix(y_true, y_pred)
    side = max(8, len(labels))
    fig, ax = plt.subplots(figsize=(side, side - 1))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {run_label}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    out_path = f"cm_{run_label}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def savePerClassBarChart(y_true, y_pred, labels, run_label):
    matrix = confusion_matrix(y_true, y_pred)
    class_scores = matrix.diagonal() / matrix.sum(axis=1)
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    bars = ax.bar(labels, class_scores, color=PLOT_PALETTE[0])
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in class_scores], padding=3, fontsize=8)
    ax.set_title(f"Per-Class Accuracy — {run_label}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.axhline(
        y=np.mean(class_scores),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(class_scores):.2f}",
    )
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = f"per_class_{run_label}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def saveTrainingTimeChart(logs, run_labels):
    minutes = [entry["training_time"] / 60 for entry in logs]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(run_labels, minutes, color=PLOT_PALETTE[: len(run_labels)])
    ax.bar_label(bars, labels=[f"{m:.1f} min" for m in minutes], padding=3)
    ax.set_title("Training Time Comparison")
    ax.set_ylabel("Time (minutes)")
    ax.set_xlabel("Model")
    plt.tight_layout()
    plt.savefig("training_times.png", dpi=150)
    plt.close()
    print("Saved: training_times.png")


def saveAccuracyComparison(summary_rows, run_labels):
    accuracies = [row["test_acc"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(run_labels, accuracies, color=PLOT_PALETTE[: len(run_labels)])
    ax.bar_label(bars, labels=[f"{a:.4f}" for a in accuracies], padding=3)
    ax.set_title("Test Accuracy — All Models")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Model")
    ax.axhline(
        y=max(accuracies),
        color="red",
        linestyle="--",
        label=f"Best: {max(accuracies):.4f}",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig("final_comparison.png", dpi=150)
    plt.close()
    print("Saved: final_comparison.png")


def transferLearningOptimizer(net, epoch_count):
    param_groups = [
        {"params": net.fc.parameters(), "lr": 1e-3},
        {
            "params": [
                p
                for name, p in net.named_parameters()
                if "fc" not in name and p.requires_grad
            ],
            "lr": 1e-4,
        },
    ]
    optimizer = optim.Adam(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return optimizer, scheduler, epoch_count


def setupCnnArch(num_classes):
    net = CNNArch(num_classes)
    print(f"CNNArch    | Total params: {countParameters(net):,}")
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    return net, optimizer, scheduler, 40


def setupMobileNetArch(num_classes):
    net = MobileNetArch(num_classes, width_mult=1.0)
    print(f"MobileNetArch | Total params: {countParameters(net):,}")
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    return net, optimizer, scheduler, 40


def setupResNetModel(num_classes):
    net = makeResNetModel(num_classes, unfreeze_blocks=2)
    optimizer, scheduler, epochs = transferLearningOptimizer(net, 25)
    return net, optimizer, scheduler, epochs


def setupGoogLeNetModel(num_classes):
    net = makeGoogLeNetModel(num_classes, unfreeze_blocks=1)
    optimizer, scheduler, epochs = transferLearningOptimizer(net, 25)
    return net, optimizer, scheduler, epochs


def Main():
    train_loader, val_loader, test_loader, num_classes, class_labels = (
        buildDataLoaders()
    )
    loss_fn = nn.CrossEntropyLoss()

    experiments = {
        "CNNArch": setupCnnArch,
        "MobileNetArch": setupMobileNetArch,
        "ResNetModel": setupResNetModel,
        "GoogLeNetModel": setupGoogLeNetModel,
    }

    all_logs, run_labels, summary = [], [], []

    for label, setup_fn in experiments.items():
        print()
        print("=================================================")
        print(f"Training: {label}")
        print("=================================================")

        net, optimizer, scheduler, max_epochs = setup_fn(num_classes)

        trained_net, run_log = fitModel(
            net,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            scheduler,
            epochs=max_epochs,
            run_label=label,
        )

        test_loss, test_acc, _, _ = evaluateOnTest(
            trained_net, test_loader, class_labels, label, loss_fn
        )

        all_logs.append(run_log)
        run_labels.append(label)
        summary.append(
            {
                "model": label,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "training_time": run_log["training_time"],
            }
        )

    print("=================================================")
    print("SUMMARY")
    print("=================================================")
    for row in summary:
        print(
            f"  {row['model']:12s} | Test Acc: {row['test_acc']:.4f} | "
            f"Time: {row['training_time'] / 60:.1f} min"
        )

    saveConvergencePlot(all_logs[:2], run_labels[:2], "convergence_scratch.png")
    saveConvergencePlot(all_logs[2:], run_labels[2:], "convergence_transfer.png")
    saveConvergencePlot(all_logs, run_labels, "convergence.png")
    saveTrainingTimeChart(all_logs, run_labels)
    saveAccuracyComparison(summary, run_labels)

    print()
    print("Done :D")


if __name__ == "__main__":
    Main()
