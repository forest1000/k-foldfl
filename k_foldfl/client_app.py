from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from sklearn.metrics import precision_score, recall_score, f1_score

from k_foldfl.task import dirichlet_partition, load_cifar10_test, make_model

# ─────────── Flower Client ───────────

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, local_epochs, lr, momentum, device):
        """
        Flower Client for federated learning with PyTorch.
        Args:
            net(torch.nn.Module): The model to train.
            trainloader(torch.utils.data.DataLoader): DataLoader for the training set.
            valloader(torch.utils.data.DataLoader): DataLoader for the validation set.
            testloader(torch.utils.data.DataLoader): DataLoader for the test set.
            local_epochs(int): Number of local training epochs.
            lr(float): Learning rate.
            momentum(float): Momentum factor for optimizer.
            device(torch.device): Device to train on (CPU or GPU).
        """
        self.net = net.to(device)
        self.trainloader, self.valloader, self.testloader = trainloader, valloader, testloader
        self.local_epochs = int(local_epochs)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=float(lr))

    def get_parameters(self, config):
        """
        サーバーにモデルパラメータを送信するための関数
        Traning終了後に実行される
        """
        return [p.detach().cpu().numpy() for p in self.net.state_dict().values()]

    def set_parameters(self, parameters):
        """
        グローバルモデルのパラメータを受け取るための関数
        ラウンドの最初に実行される
        """
        sd = self.net.state_dict()
        for k, v in zip(sd.keys(), parameters):
            sd[k] = torch.tensor(v, device=self.device)
        self.net.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        if config.get("early_stop", False):
            return self.get_parameters(config), len(self.trainloader.dataset), {}
        
        self.net.train()
        for _ in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(x), y)
                loss.backward(); 
                self.optimizer.step()
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Test the model on the validation or test set.
        Phase is decided on server side.
        Args:
            parameters (list): Model parameters to evaluate.
            config (dict): Configuration for evaluation. You can treat this config in server-side logic.

        Return:
            tuple: A tuple containing the following elements:
                - float: The average loss on the dataset.
                - int: The number of samples in the dataset.
                - dict: A dictionary containing additional metrics (e.g., accuracy, precision, recall, F1 score).

        """
        self.set_parameters(parameters)
        phase = config.get("phase", "val")  # "val" | "test"
        loader = self.testloader if phase == "test" else self.valloader

        self.net.eval()
        total, correct = 0, 0
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.net(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * y.size(0)
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(y.detach().cpu().tolist())

        # 評価したい指標
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1        = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        n = len(loader.dataset)
        # 注意点として、送信する値はPythonの標準の方である必要がある：シリアライズのため
        return float(avg_loss), int(n), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "phase": phase,
        }

# ─────────── ClientApp factory ───────────

def make_client_app(full_ds, fold_cfg, parts, train_idx, val_idx):
    exp = fold_cfg["experiment"]
    bs = int(exp["batch_size"])
    local_epochs = int(exp["local_epochs"])
    lr = float(exp["lr"])
    momentum = float(exp["momentum"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = exp.get("model", "resnet18")

    
    def client_fn(context: Context):
        net = make_model(model_name)
        partition_id = int(context.node_config["partition-id"])
        local_train_idx = parts[partition_id]

        trainloader = DataLoader(
            Subset(full_ds, local_train_idx),
            batch_size=bs, shuffle=True, num_workers=0
        )
        valloader = DataLoader(
            Subset(full_ds, val_idx),
            batch_size=bs, shuffle=False, num_workers=0
        )
        test_ds = load_cifar10_test(exp["dataset_root"])
        testloader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

        client = FlowerClient(net, trainloader, valloader, testloader, local_epochs, lr, momentum, device)
        return client.to_client()

    return ClientApp(client_fn=client_fn)