import argparse
import os
import random
from collections import defaultdict

import torch
import torch.optim as optim
from tqdm import tqdm

from experiment_configs import EXPERIMENTS, HYPERPARAMETERS
from result_metrics import evaluate_across_time, evaluate_rps
from saving_results import (
    load_checkpoint,
    make_run_dir,
    plot_training_curves,
    save_checkpoint,
)


def group_indices_by_match(dataset):
    match_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        match_id = dataset.get(idx).match_id
        match_to_indices[match_id].append(idx)
    return match_to_indices


def split_by_match(dataset, train_ratio=0.7, seed=42):
    val_ratio = (1 - train_ratio) / 2
    random.seed(seed)
    match_to_indices = group_indices_by_match(dataset)

    all_matches = list(match_to_indices.keys())
    random.shuffle(all_matches)

    n_total = len(all_matches)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_matches = all_matches[:n_train]
    val_matches = all_matches[n_train : n_train + n_val]
    test_matches = all_matches[n_train + n_val :]

    train_indices = [i for m in train_matches for i in match_to_indices[m]]
    val_indices = [i for m in val_matches for i in match_to_indices[m]]
    test_indices = [i for m in test_matches for i in match_to_indices[m]]

    return train_indices, val_indices, test_indices


def train_one_epoch(model, dataset, criterion, optimizer, device, forward_pass):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    tqdm_dataset = tqdm(dataset, desc="Training", leave=False)
    for entry in tqdm_dataset:
        optimizer.zero_grad()

        out, y, home_goals, away_goals = forward_pass(entry, model, device)

        loss = criterion(out, y, home_goals, away_goals)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out["class_logits"].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        tqdm_dataset.set_postfix(loss=f"{loss.item():.4f}")

    accuracy = 100 * correct / total if total > 0 else 0
    return total_loss / len(dataset), accuracy


@torch.no_grad()
def evaluate(model, dataset, criterion, device, forward_pass):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    tqdm_dataset = tqdm(dataset, desc="Evaluating", leave=False)
    for entry in tqdm_dataset:

        out, y, home_goals, away_goals = forward_pass(entry, model, device)

        loss = criterion(out, y, home_goals, away_goals)
        total_loss += loss.item()

        preds = out["class_logits"].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    accuracy = 100 * correct / total if total > 0 else 0
    return total_loss / len(dataset), accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default="small",
        choices=EXPERIMENTS.keys(),
        help="Experiment configuration to run",
    )
    parser.add_argument(
        "--retrain-model",
        action="store_true",
        help="Whether you want to retrain an already trained model, if there is one. This will redo any training and overwrite the current saved model. Otherwise, the saved model will be used to evaluate.",
    )
    args = parser.parse_args()

    cfg = EXPERIMENTS[args.exp]
    hyp = HYPERPARAMETERS
    print(f"Running experiment: {cfg.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = cfg.dataset_factory()
    print(f"Dataset size: {len(dataset)}")

    forward_pass = cfg.forward_pass

    # Split into train/test
    train_idx, val_idx, test_idx = split_by_match(
        dataset, train_ratio=cfg.train_split, seed=cfg.seed
    )

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # Model setup
    model = cfg.model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    run_dir, exists = make_run_dir(cfg, hyp)
    print(f"Run directory: {run_dir}")

    should_load_model = not args.retrain_model and exists

    criterion = cfg.criterion
    optimizer = optim.Adam(
        model.parameters(), lr=hyp.learning_rate, weight_decay=hyp.weight_decay
    )

    num_epochs = hyp.num_epochs
    start_epoch, best_acc, best_test_loss, history = load_checkpoint(
        model, optimizer, run_dir
    )

    early_stopping_counter = 0

    # === TRAINING LOOP ===================================================
    if not should_load_model:
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = train_one_epoch(
                model, train_dataset, criterion, optimizer, device, forward_pass
            )
            test_loss, test_acc = evaluate(
                model, test_dataset, criterion, device, forward_pass
            )

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
                print(f"🏆 New best model saved (val_loss={test_loss:.4f})")
            else:
                early_stopping_counter += 1
                print(
                    f"No improvement for {early_stopping_counter}/{hyp.patience} epochs"
                )

            save_checkpoint(
                model, optimizer, epoch, best_acc, best_test_loss, history, run_dir
            )

            print(
                f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%"
            )

            if early_stopping_counter >= hyp.patience:
                print("🛑 Early stopping triggered.")
                break
            plot_training_curves(history, run_dir)

        print("\n✅ Training complete!")
        print(f"Best test loss: {best_test_loss:.4f}")
        print("Validation test:")
        val_loss, val_acc = evaluate(
            model, val_dataset, criterion, device, forward_pass
        )
        print(f"Validation loss: {val_loss}")
        print(f"Validation accuracy: {val_acc}")
        print(f"Run directory: {run_dir}")
    else:
        ###WHAT WE WANT TO DO, STUDIES
        evaluate_rps(model, test_dataset, device, forward_pass)
        evaluate_across_time(model, test_dataset, device, forward_pass, run_dir)


if __name__ == "__main__":
    main()
