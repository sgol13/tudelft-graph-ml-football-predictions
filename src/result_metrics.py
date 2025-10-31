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
def evaluate_across_time(model, dataloader, device, forward_pass):
    model.eval()

    correct_by_time = {}
    total_by_time = {}

    progress = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress:
        # Forward pass
        out, y, _, _ = forward_pass(batch, model, device)
        preds = out["class_logits"].argmax(dim=1)

        # Extraemos todos los end_minutes de este batch
        end_minutes = batch["sequences"][-1].end_minute

        # Aseguramos que esté en CPU y tipo lista
        if torch.is_tensor(end_minutes):
            end_minutes = end_minutes.cpu().tolist()

        # Recorremos cada muestra dentro del batch
        for minute, pred, label in zip(end_minutes, preds, y):
            minute = int(minute)

            if minute not in correct_by_time:
                correct_by_time[minute] = 0
                total_by_time[minute] = 0

            correct_by_time[minute] += int(pred == label)
            total_by_time[minute] += 1

    # Calcular accuracies
    times = sorted(correct_by_time.keys())
    accuracies = [correct_by_time[t] / total_by_time[t] for t in times]

    plt.plot(times, accuracies, marker='o')
    plt.xlabel("Minute")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Time")
    plt.grid(True)
    plt.show()

    return times, accuracies

