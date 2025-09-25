from typing import Dict, List, Tuple
import numpy as np
import flwr as fl
from flwr.common import Context, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents
from flwr.server.client_proxy import ClientProxy
from k_foldfl.task import get_weights, make_model
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import OrderedDict

# ─────────── Metrics aggregation（重み付き平均） ───────────
# 集約時の手法の改良をしたいなら、この部分を変更する
def weighted_metrics_agg(results: List[Tuple[Dict[str, float], int]]) -> Dict[str, float]:
    if not results:
        return {}
    sums = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_tot = 0
    for n, metrics in results:
        n_tot += n
        for k in sums.keys():
            v = float(metrics.get(k, 0.0))
            sums[k] += n * v
    return {k: (sums[k] / max(n_tot, 1)) for k in sums.keys()}

# ─────────── Strategy ───────────
class SaveMetricsFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        num_rounds: int,
        base_local_epochs: int,
        monitor: str = "accuracy",
        mode: str = "max",
        patience: int = 3,
        min_delta: float = 0.0,
        *,
        test_root: str,
        batch_size: int,
        device: torch.device,
        model_name: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_rounds = int(num_rounds)
        self.base_local_epochs = int(base_local_epochs)
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.round_metrics: List[Dict[str, float]] = []
        self.early_stop = False
        self.bad_cnt = 0
        self.best = -np.inf if mode == "max" else np.inf
        self.best_round = None
        self.best_parameters = None          # list[np.ndarray] or Parameters を想定
        self.best_metrics: Dict[str, float] = {}
        self._last_agg_params = None         # list[np.ndarray]（aggregate_fitで更新）
        self.final_server_metrics: Dict[str, float] = {}

        # --- サーバーTest用の準備 ---
        self._device = device
        self._model_name = model_name
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])
        test_ds = datasets.CIFAR10(test_root, train=False, download=True, transform=tfm)
        self._testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        self._criterion = torch.nn.CrossEntropyLoss()

    # 学習指示：早期終了後は local_epochs=0 / early_stop=True
    def configure_fit(self, server_round, parameters, client_manager):
        base = super().configure_fit(server_round, parameters, client_manager)
        new_ins = []
        for client, fitins in base:
            cfg = dict(fitins.config)
            if self.early_stop:
                cfg["early_stop"] = True
                cfg["local_epochs"] = 0
            else:
                cfg.setdefault("early_stop", False)
                cfg.setdefault("local_epochs", self.base_local_epochs)
            new_ins.append((client, fl.common.FitIns(fitins.parameters, cfg)))
        return new_ins

    # 直近の集約重みを NumPy リストで保持
    def aggregate_fit(self, server_round, results, failures):
        res = super().aggregate_fit(server_round, results, failures)
        if res is not None:
            params, _ = res
            self._last_agg_params = parameters_to_ndarrays(params)  # list[np.ndarray]
        return res

    # 評価指示：常に各クライアントで validation（最終ラウンドでも "val"）
    def configure_evaluate(self, server_round, parameters, client_manager):
        cfg = {"phase": "val"}
        eval_ins = fl.common.EvaluateIns(parameters, cfg)
        clients: List[ClientProxy] = list(client_manager.all().values())
        return [(client, eval_ins) for client in clients]

    # 分散validationの集約＋早期停止判定＋（最終ラウンドなら）サーバーTest
    def aggregate_evaluate(self, server_round, results, failures):
        res = super().aggregate_evaluate(server_round, results, failures)
        if res is None:
            return None

        loss, metrics = res
        if isinstance(metrics, dict) and metrics:
            self.round_metrics.append({"loss": float(loss) if loss is not None else None, **metrics})

            # 監視値で早期停止の更新
            val = float(loss) if self.monitor == "loss" and loss is not None else float(metrics.get(self.monitor, np.nan))
            if not np.isnan(val):
                improved = (val > self.best + self.min_delta) if self.mode == "max" else (val < self.best - self.min_delta)
                if improved:
                    self.best = val
                    self.bad_cnt = 0
                    self.best_round = server_round
                    self.best_metrics = metrics
                    self.best_parameters = self._last_agg_params  # この評価に対応する最新重み（list[np.ndarray]）
                else:
                    self.bad_cnt += 1
                    if self.bad_cnt >= self.patience and server_round < self.num_rounds:
                        self.early_stop = True  # 次ラウンド以降のfitをスキップ

        # 最終ラウンドの集約が終わった“後”に、サーバーTest を実行
        if server_round == self.num_rounds:
            self._run_server_test()

        return res

    # --- サーバーTest（best があれば best、無ければ直近集約重み） ---
    def _run_server_test(self):
        params_to_use = self.best_parameters if self.best_parameters is not None else self._last_agg_params
        if params_to_use is None:
            return  # 何もなければ実行不可

        nds = list(params_to_use)
        model = make_model(self._model_name).to(self._device)
        sd_keys = list(model.state_dict().keys())
        state_dict = OrderedDict({k: torch.tensor(v, device=self._device) for k, v in zip(sd_keys, nds)})
        model.load_state_dict(state_dict, strict=True)

        total, correct, loss_sum = 0, 0, 0.0
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for x, y in self._testloader:
                x, y = x.to(self._device), y.to(self._device)
                logits = model(x)
                loss = self._criterion(logits, y)
                loss_sum += loss.item() * y.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        avg_loss = loss_sum / max(total, 1)
        acc  = correct / max(total, 1)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        self.final_server_metrics = {
            "server_test_loss": float(avg_loss),
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
        }
        print(f"[Server] FINAL centralized TEST (best params) "
              f"loss={avg_loss:.4f} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

# ─────────── ServerApp factory ───────────
def make_server_app(fold_cfg):
    exp = fold_cfg["experiment"]; st = fold_cfg["strategy"]
    n_clients = int(exp["num_clients"])
    n_rounds  = int(exp["rounds"])
    base_local_epochs = int(exp["local_epochs"])
    model_name  = exp.get("model", "resnet18")
    dataset_root = exp["dataset_root"]
    batch_size = int(exp["batch_size"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    patience  = int(st.get("patience", 3))
    min_delta = float(st.get("min_delta", 0.0))
    monitor   = st.get("monitor", "accuracy")
    mode      = st.get("mode", "max")

    model = make_model(model_name)

    strategy = SaveMetricsFedAvg(
        num_rounds=n_rounds,
        base_local_epochs=base_local_epochs,
        monitor=monitor, mode=mode,
        patience=patience, min_delta=min_delta,
        fraction_fit=float(st["fraction_fit"]),
        fraction_evaluate=float(st["fraction_evaluate"]),
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=max(n_clients, int(st["min_available_clients"])),
        initial_parameters=fl.common.ndarrays_to_parameters(get_weights(model)),
        test_root=dataset_root,
        batch_size=batch_size,
        device=device,
        model_name=model_name,
    )

    server_cfg = fl.server.ServerConfig(num_rounds=n_rounds)

    def server_fn(context: Context):
        return ServerAppComponents(config=server_cfg, strategy=strategy)

    return ServerApp(server_fn=server_fn), strategy
