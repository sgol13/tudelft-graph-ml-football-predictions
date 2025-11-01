import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def ranked_probability_score(pred_probs: torch.Tensor, true_onehot: torch.Tensor):
    """
    pred_probs: [batch_size, num_classes], probabilities for each class
    true_onehot: [batch_size, num_classes], one-hot encoded true labels
    """
    # Cumulative sums along classes
    F = torch.cumsum(pred_probs, dim=1)
    O = torch.cumsum(true_onehot, dim=1)

    # Squared differences
    rps = torch.mean(torch.sum((F - O) ** 2, dim=1))  # mean over batch
    return rps.item()

@torch.no_grad()
def evaluate_rps(model, dataloader, device, forward_pass):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        out, y, _, _ = forward_pass(batch, model, device)
        
        probs = torch.softmax(out["class_logits"], dim=1)
        
        all_preds.append(probs.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Convert labels to one-hot if needed
    num_classes = all_preds.shape[1]
    labels_onehot = torch.nn.functional.one_hot(all_labels, num_classes=num_classes).float()

    rps_score = ranked_probability_score(all_preds, labels_onehot)
    print("RPS:", rps_score)



@torch.no_grad()
def evaluate_across_time(model, dataloader, device, forward_pass, run_dir):
    model.eval()
    correct_by_minute = {}
    total_by_minute = {}

    progress = tqdm(dataloader, desc="Evaluating across time", leave=False)
    for batch in progress:
        out, y, _, _ = forward_pass(batch, model, device)
        preds = out["class_logits"].argmax(dim=1)

        # 🔹 obtener end_minute directamente del batch
        end_minutes = [
            seq[-1].end_minute.item()
            for seq in batch["sequences"]
        ]

        for pred, label, end_min in zip(preds, y, end_minutes):
            end_min = int(end_min)
            correct_by_minute[end_min] = correct_by_minute.get(end_min, 0) + (pred == label).item()
            total_by_minute[end_min] = total_by_minute.get(end_min, 0) + 1

    # calcular accuracy por minuto
    accuracy_by_minute = {
        m: 100 * correct_by_minute[m] / total_by_minute[m]
        for m in sorted(total_by_minute.keys())
    }



    plt.plot(list(accuracy_by_minute.keys()), list(accuracy_by_minute.values()), marker="o")
    plt.xlabel("End minute")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Time Frame")
    plt.savefig(f"{run_dir}/accuracy_across_time.png")

    return accuracy_by_minute



