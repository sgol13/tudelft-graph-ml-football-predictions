import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tabulate import tabulate
from tqdm import tqdm


def extract_time_interval(path: str) -> int | None:
    match = re.search(r"time_interval(\d+)", path)
    return int(match.group(1)) if match else None


def compute_rps(preds_probs, y_true):
    """
    Ranked Probability Score (RPS) for classification.
    Handles both 1D (single sample) and 2D (batch) inputs.
    preds_probs: Tensor [num_classes] or [N, num_classes]
    y_true: Tensor [] or [N] with class indices
    """
    # Ensure both are tensors on the same device
    if not torch.is_tensor(preds_probs):
        preds_probs = torch.tensor(preds_probs)
    device = preds_probs.device
    y_true = y_true.to(device)

    # Add batch dimension if missing
    if preds_probs.ndim == 1:
        preds_probs = preds_probs.unsqueeze(0)
    if y_true.ndim == 0:
        y_true = y_true.unsqueeze(0)

    num_classes = preds_probs.size(1)
    y_true_onehot = torch.zeros_like(preds_probs).scatter_(1, y_true.unsqueeze(1), 1)

    cum_probs = torch.cumsum(preds_probs, dim=1)
    cum_true = torch.cumsum(y_true_onehot, dim=1)
    rps = torch.mean(torch.sum((cum_probs - cum_true) ** 2, dim=1)) / (num_classes - 1)
    return rps.item()


@torch.no_grad()
def evaluate_plus(
    model, dataset, criterion, device, forward_pass, run_dir, is_cumulative
):
    """
    Evaluates the model over the dataset, supporting variable-length y.

    Args:
        is_cumulative: If True, treats dataset as CumulativeDataset where each entry
                      has a single prediction at end_minute. Filters to intervals of 5
                      and maps to positions (end_minute // 5).

    Returns:
      - total accuracy (%)
      - total RPS
      - per-position accuracy and RPS (aggregated across all entries that had that position)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_rps = 0.0
    total_samples = 0

    # Dynamic per-position stats (indexed by position number)
    per_pos_correct = {}
    per_pos_total = {}
    per_pos_rps = {}

    tqdm_dataset = tqdm(dataset, desc="Evaluating +", leave=False)

    for entry in tqdm_dataset:
        # For CumulativeDataset, filter to only process entries at 5-minute intervals
        if is_cumulative:
            end_minute = entry.end_minute.item()

            # Calculate position based on end_minute
            position = max((end_minute // 5) - 1, 0)

        out, y, home_goals, away_goals = forward_pass(entry, model, device)

        loss = criterion(out, y, home_goals, away_goals)
        total_loss += loss.item()

        probs = torch.softmax(out["class_logits"], dim=-1)
        preds = probs.argmax(dim=1)

        if is_cumulative:
            # For cumulative dataset, y is a single value
            # Ensure y is 0-dimensional or has single element

            correct = float(preds == y)
            rps = compute_rps(probs, y)

            # Accumulate global stats
            total_correct += correct
            total_rps += rps
            total_samples += 1

            # Accumulate per-position stats using calculated position
            if position not in per_pos_correct:
                per_pos_correct[position] = 0.0
                per_pos_total[position] = 0.0
                per_pos_rps[position] = 0.0

            per_pos_correct[position] += correct
            per_pos_total[position] += 1
            per_pos_rps[position] += rps
        else:
            n_pos = y.numel()

            for i in range(n_pos):
                yi = y[i]
                probs_i = probs[i]
                pred_i = preds[i]

                correct_i = float(pred_i == yi)
                rps_i = compute_rps(probs_i, yi)

                # Accumulate global stats
                total_correct += correct_i
                total_rps += rps_i
                total_samples += 1

                # Accumulate per-position stats
                if i not in per_pos_correct:
                    per_pos_correct[i] = 0.0
                    per_pos_total[i] = 0.0
                    per_pos_rps[i] = 0.0

                per_pos_correct[i] += correct_i
                per_pos_total[i] += 1
                per_pos_rps[i] += rps_i

    # === Aggregate results ===
    total_acc = 100 * total_correct / total_samples if total_samples > 0 else 0
    total_rps /= total_samples if total_samples > 0 else 1

    per_position = []
    for i in sorted(per_pos_correct.keys()):
        acc_i = 100 * per_pos_correct[i] / per_pos_total[i]
        rps_i = per_pos_rps[i] / per_pos_total[i]
        per_position.append({"pos": i, "acc": acc_i, "rps": rps_i})

    results = {
        "loss": total_loss / len(dataset),
        "accuracy": total_acc,
        "rps": total_rps,
        "per_position": per_position,
    }

    print(f"Total Accuracy: {results['accuracy']:.2f}%")
    print(f"Total RPS: {results['rps']:.4f}")
    print("Per position:")
    for p in results["per_position"]:
        print(f"  Pos {p['pos']}: Acc={p['acc']:.2f}%, RPS={p['rps']:.4f}")

    os.makedirs(run_dir, exist_ok=True)
    save_path = os.path.join(run_dir, "evaluate_plus_results.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


def compare_models(metrics_paths, save_dir=None):
    """
    Compare evaluation_plus results across multiple models.

    Args:
        metrics_paths (dict): {model_name: path_to_json}
        time_interval (int): the length of every window
        save_dir (str, optional): where to save the comparison plots. If None, no saving.

    Each JSON is expected to contain:
        {
            "loss": float,
            "accuracy": float,
            "rps": float,
            "per_position": [
                {"pos": int, "acc": float, "rps": float}, ...
            ],
        }
    """
    data = {}
    time_interval = {}
    # === Load JSON files ===
    for name, path in metrics_paths.items():
        if not os.path.exists(path):
            print(f"⚠️  Missing file for {name}: {path}")
            continue
        with open(path, "r") as f:
            data[name] = json.load(f)
            time_interval[name] = extract_time_interval(path)

    if not data:
        print("❌ No valid model results found.")
        return

    # === Print comparison table ===
    table = []
    for name, res in data.items():
        table.append([name, res["loss"], res["accuracy"], res["rps"]])

    print("\n📊 Model Comparison Summary:")
    print(
        tabulate(
            table, headers=["Model", "Loss", "Accuracy (%)", "RPS"], floatfmt=".4f"
        )
    )

    # === Plot accuracy and RPS per position ===
    plt.figure(figsize=(10, 4))
    for name, res in data.items():
        xs = [(p["pos"] + 1) * time_interval[name] for p in res["per_position"]]
        accs = [p["acc"] for p in res["per_position"]]
        plt.plot(xs, accs, marker="o", label=name)
    plt.title("Accuracy per Minute")
    plt.xlabel("Time")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        acc_path = os.path.join(save_dir, "compare_accuracy.png")
        plt.savefig(acc_path, bbox_inches="tight")
        print(f"📈 Saved accuracy plot to: {acc_path}")
    plt.show()

    plt.figure(figsize=(10, 4))
    for name, res in data.items():
        xs = [(p["pos"] + 1) * time_interval[name] for p in res["per_position"]]
        rpss = [p["rps"] for p in res["per_position"]]
        plt.plot(xs, rpss, marker="o", label=name)
    plt.title("RPS per Minute")
    plt.xlabel("Time")
    plt.ylabel("RPS")
    plt.legend()
    plt.grid(True)
    if save_dir:
        rps_path = os.path.join(save_dir, "compare_rps.png")
        plt.savefig(rps_path, bbox_inches="tight")
        print(f"📉 Saved RPS plot to: {rps_path}")
    plt.show()


def main():
    MODELS = {
        "VARMA": f"{Path.cwd().as_posix()}/runs/2020_2024/varma/time_interval5/goal_False/lr0.0005_wr1e-05_a1.0_b0.5/evaluate_plus_results.json",
        "RNN": f"{Path.cwd().as_posix()}/runs/2020_2024/rnn/time_interval5/goal_False/lr0.0005_wr1e-05_a1.0_b0.5/evaluate_plus_results.json",
        "GAT": f"{Path.cwd().as_posix()}/runs/2020_2024/large/time_interval5/goal_False/lr0.0005_wr1e-05_a1.0_b0.5/evaluate_plus_results.json",
        "Disjoint": f"{Path.cwd().as_posix()}/runs/2020_2024/disjoint/time_interval5/goal_False/lr0.0005_wr1e-05_a1.0_b0.5/evaluate_plus_results.json",
    }
    compare_models(MODELS, "plots/comparison_all_models_ce_5min")

    MODELS_DISJOINT = {
        "goal_loss": f"{Path.cwd().as_posix()}/runs/2020_2024/disjoint/time_interval5/goal_True/lr0.0005_wr1e-05_a0.1_b1/evaluate_plus_results.json",
        "ce + goal_loss": f"{Path.cwd().as_posix()}/runs/2020_2024/disjoint/time_interval5/goal_True/lr0.0005_wr1e-05_a1.0_b0.5/evaluate_plus_results.json",
        "ce": f"{Path.cwd().as_posix()}/runs/2020_2024/disjoint/time_interval5/goal_False/lr0.0005_wr1e-05_a1.0_b0.5/evaluate_plus_results.json",
    }
    compare_models(MODELS_DISJOINT, "plots/loss")

    MODELS_INTERVAL = {
        "5": f"{Path.cwd().as_posix()}/runs/2020_2024/disjoint/time_interval5/goal_True/lr0.0005_wr1e-05_a1.0_b0.5/evaluate_plus_results.json",
        "9": f"{Path.cwd().as_posix()}/runs/2020_2024/disjoint/time_interval9/goal_True/lr0.0005_wr1e-05_a1.0_b0.5/evaluate_plus_results.json",
        "15": f"{Path.cwd().as_posix()}/runs/2020_2024/disjoint/time_interval15/goal_True/lr0.0005_wr1e-05_a1.0_b0.5/evaluate_plus_results.json",
    }
    compare_models(MODELS_INTERVAL, "plots/interval")


if __name__ == "__main__":
    main()
