import os, random, warnings
import numpy as np
import csv, datetime
import flwr as fl
import torch
from args_parser import args, args2cfg
from sklearn.model_selection import KFold

from k_foldfl.client_app import make_client_app
from k_foldfl.server_app import make_server_app
from k_foldfl.task import load_yaml, load_cifar10, dirichlet_partition

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def avg_metrics_across_folds(metrics_list):
    if not metrics_list: return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m.get(k, 0.0) for m in metrics_list])) for k in keys}

def main(cfg_path="configs/config.yaml"):
    base_cfg = args2cfg(load_yaml(cfg_path), args)
    exp = base_cfg["experiment"]
    set_seed(int(exp["seed"]))
    full_ds = load_cifar10(exp["dataset_root"])

    kf = KFold(n_splits=int(exp["k_folds"]), shuffle=True, random_state=int(exp["seed"]))
    all_final_metrics = []
    csv_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_ds)))):
        fold_cfg = base_cfg
        print(f"\n===== Fold {fold_id+1}/{exp['k_folds']} =====")
        
        # Dirichlet で fold の学習部分（train_idx）を num_partitions に分割
        beta = float(fold_cfg["experiment"]["beta"])
        num_partitions = int(fold_cfg["experiment"]["num_clients"])
        seed = int(fold_cfg["experiment"].get("seed", 42))
        parts = dirichlet_partition(train_idx, full_ds.targets, num_partitions, beta, seed)

        client_app = make_client_app(full_ds, fold_cfg, parts, train_idx.tolist(), val_idx.tolist())
        server_app, strategy = make_server_app(fold_cfg)

        client_res = fold_cfg["resources"]["client"]
        backend_config = {
            "client_resources": {
                "num_cpus": float(client_res.get("num_cpus", 1)),
                "num_gpus": float(client_res.get("num_gpus", 0)),
            },
            "ray_init_args": {"include_dashboard": bool(base_cfg["backend"].get("include_dashboard", False))}
        }

        fl.simulation.run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=int(fold_cfg["experiment"]["num_clients"]),
            backend_config=backend_config,
        )

        final_metrics = strategy.round_metrics[-1] if strategy.round_metrics else {}
        print(f"Fold {fold_id+1} final (TEST, client-agg): {final_metrics}")
        all_final_metrics.append(final_metrics)

        #サーバTEST の結果（best params）を CSV 行に入れる
        server_m = getattr(strategy, "final_server_metrics", {})  # 例: {"server_test_loss":..., "test_accuracy":...}

        row = {"fold": fold_id + 1}

        # サーバ TEST（best 重み）

        row["server_test_loss"]      = float(server_m.get("server_test_loss", np.nan))
        row["server_test_accuracy"]  = float(server_m.get("server_test_accuracy", server_m.get("test_accuracy", np.nan)))
        row["server_test_precision"] = float(server_m.get("server_test_precision", server_m.get("test_precision", np.nan)))
        row["server_test_recall"]    = float(server_m.get("server_test_recall", server_m.get("test_recall", np.nan)))
        row["server_test_f1"]        = float(server_m.get("server_test_f1", server_m.get("test_f1", np.nan)))

        csv_rows.append(row)

        # ログ表示
        if server_m:
            print(f"Fold {fold_id+1} server-side FINAL TEST (best): {server_m}")
        else:
            print(f"Fold {fold_id+1} server-side FINAL TEST (best): <no result>")

    print("\n========== 平均 (TEST: client-agg) ==========")
    avg = avg_metrics_across_folds(all_final_metrics)  
    for k, v in avg.items():
        print(f"{k:>10}: {v:.4f}")

    # ---- CSV 保存 ---

    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join("results", f"kfold_test_metrics_{ts}.csv")

   
    fieldnames = [
        "fold",
        "accuracy", "precision", "recall", "f1",
        "server_test_loss", "server_test_accuracy", "server_test_precision",
        "server_test_recall", "server_test_f1",
    ]

   
    mean_row = {"fold": "mean"}
    for k in fieldnames[1:]:
        vals = [r[k] for r in csv_rows if (k in r and isinstance(r[k], (int, float)) and not np.isnan(r[k]))]
        mean_row[k] = float(np.mean(vals)) if vals else np.nan

    rows_to_write = csv_rows + [mean_row]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(f"\nSaved CSV: {csv_path}")
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()