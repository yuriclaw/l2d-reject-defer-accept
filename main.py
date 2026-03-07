"""
Learning to Defer with Three Actions: Accept / Defer / Reject
Based on the three-logits surrogate loss from the paper.

Actions:
  1. Predict using local model m(x)
  2. Predict using expert model e(x)  (defer)
  3. Abstain / Reject

Loss:
  L = 1{r(x)=1} * 1{m(x)!=y} + 1{r(x)=2} * (1{e(x)!=y} + c1(s)) + 1{r(x)=3} * c

Where c1(s) is the variable deferral cost depending on system status s,
and c is the fixed abstention cost.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import os


# ============================================================
# Models
# ============================================================

class LocalModel(nn.Module):
    """Simple CNN as the local (weak) model."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ExpertModel(nn.Module):
    """Stronger model simulating a cloud expert (ResNet-18)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = torchvision.models.resnet18(num_classes=num_classes)
        # Adjust for CIFAR-10 (32x32 images)
        self.net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


class Rejector(nn.Module):
    """Three-logit rejector: outputs scores for [accept, defer, reject]."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 3),  # 3 actions
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# ============================================================
# Loss Functions
# ============================================================

class ThreeLogitSurrogateLoss(nn.Module):
    """
    Surrogate loss for the three-action rejector.
    
    For each sample:
      - action 1 (accept): cost = 1{m(x) != y}  (local model error)
      - action 2 (defer):  cost = 1{e(x) != y} + c1(s)  (expert error + deferral cost)
      - action 3 (reject): cost = c  (fixed abstention cost)
    
    We use cross-entropy with cost-sensitive targets.
    """
    def __init__(self, abstain_cost=0.3):
        super().__init__()
        self.abstain_cost = abstain_cost

    def forward(self, rejector_logits, local_logits, expert_preds, targets, deferral_cost):
        """
        Args:
            rejector_logits: (B, 3) - logits for [accept, defer, reject]
            local_logits: (B, K) - RAW logits from local model (differentiable!)
            expert_preds: (B,) - predicted labels from expert (no grad needed)
            targets: (B,) - true labels
            deferral_cost: scalar or (B,) - variable cost c1(s)
        """
        batch_size = rejector_logits.size(0)
        
        # Local model error: use soft probability (differentiable)
        # η_m(x) = P(M=Y|X=x), estimated via softmax
        local_probs = torch.softmax(local_logits, dim=1)
        # Gather the probability of the true class
        eta_m = local_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        local_error = 1.0 - eta_m  # soft local error, gradients flow to local model
        
        # Expert error: hard (expert is frozen, no grad needed)
        expert_error = (expert_preds != targets).float()
        
        # Cost vector for each sample: [accept_cost, defer_cost, reject_cost]
        costs = torch.stack([
            local_error,
            expert_error + deferral_cost,
            torch.full((batch_size,), self.abstain_cost, device=rejector_logits.device),
        ], dim=1)  # (B, 3)
        
        # Surrogate: softmax cross-entropy weighted by costs
        # Minimize expected cost: sum_a P(a|x) * cost(a)
        probs = torch.softmax(rejector_logits, dim=1)
        loss = (probs * costs).sum(dim=1).mean()
        
        return loss


class PostHocRejector:
    """
    Post-hoc threshold adjustment for an already-trained L2D system.
    Adjusts the deferral threshold based on current system status c1(s).
    
    Bayes rule (post-hoc):
      - Accept if η_m(x) > max{1-c, η_e(x)}
      - Defer  if η_e(x) > max{1-c, η_m(x)}  AND c1 is low enough
      - Reject if η_m(x) ≤ 1-c+c1 AND η_e(x) ≤ 1-c+c1
    """
    def __init__(self, abstain_cost=0.3):
        self.abstain_cost = abstain_cost

    def decide(self, local_probs, expert_probs, deferral_cost):
        """
        Args:
            local_probs: (B, K) softmax outputs of local model
            expert_probs: (B, K) softmax outputs of expert model
            deferral_cost: scalar - current system deferral cost c1
        Returns:
            actions: (B,) - 0=accept, 1=defer, 2=reject
        """
        eta_m = local_probs.max(dim=1).values   # max local confidence
        eta_e = expert_probs.max(dim=1).values   # max expert confidence
        
        c = self.abstain_cost
        threshold = 1 - c + deferral_cost
        
        actions = torch.full((local_probs.size(0),), 2, dtype=torch.long,
                             device=local_probs.device)  # default: reject
        
        # Defer if expert is confident enough and better than local
        defer_mask = (eta_e > (1 - c)) & (eta_e > eta_m)
        actions[defer_mask] = 1
        
        # Accept if local is confident enough and better than expert
        accept_mask = (eta_m > (1 - c)) & (eta_m > eta_e)
        actions[accept_mask] = 0
        
        # Override: reject if both below adjusted threshold
        reject_mask = (eta_m <= threshold) & (eta_e <= threshold)
        actions[reject_mask] = 2
        
        return actions


# ============================================================
# Deferral Cost Simulation
# ============================================================

def simulate_deferral_cost(num_clients, base_latency=0.01, queue_factor=0.05):
    """
    Simulate variable deferral cost based on number of active clients.
    Models queue buildup: cost increases with more clients deferring.
    
    c1(s) = base_latency + queue_factor * num_active_deferrals
    """
    return base_latency + queue_factor * num_clients


# ============================================================
# Training
# ============================================================

def train_models(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # Models
    local_model = LocalModel().to(device)
    expert_model = ExpertModel().to(device)
    rejector = Rejector().to(device)
    
    # Phase 1: Train expert model separately (cloud server, pre-trained)
    print("\n=== Phase 1: Training Expert Model (standalone) ===")
    train_classifier(expert_model, trainloader, testloader, device,
                     epochs=args.expert_epochs, lr=0.001, name="Expert")
    
    # Phase 2: Joint training of Local Model + Rejector
    # In standard L2D, expert is fixed; local model and rejector are trained
    # jointly with the same loss function.
    print("\n=== Phase 2: Joint Training (Local Model + Rejector) ===")
    expert_model.eval()
    
    # Joint optimizer for both local model and rejector
    joint_optimizer = optim.Adam(
        list(local_model.parameters()) + list(rejector.parameters()), lr=0.001
    )
    loss_fn = ThreeLogitSurrogateLoss(abstain_cost=args.abstain_cost)
    ce_loss = nn.CrossEntropyLoss()
    alpha = args.ce_weight  # weight for auxiliary CE loss
    
    for epoch in range(args.joint_epochs):
        local_model.train()
        rejector.train()
        total_loss = 0
        correct, total = 0, 0
        for images, labels in tqdm(trainloader, desc=f"Joint Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass: local model produces logits (kept in graph!)
            local_logits = local_model(images)
            
            with torch.no_grad():
                expert_preds = expert_model(images).argmax(dim=1)
            
            # Simulate variable deferral cost
            num_clients = np.random.randint(1, args.max_clients + 1)
            deferral_cost = simulate_deferral_cost(num_clients)
            
            # Surrogate loss for system-level optimization
            rej_logits = rejector(images)
            surr_loss = loss_fn(rej_logits, local_logits, expert_preds, labels, deferral_cost)
            
            # Auxiliary CE loss to maintain local model classification ability
            local_ce = ce_loss(local_logits, labels)
            
            # Combined: surrogate (system optimization) + CE (classification baseline)
            loss = surr_loss + alpha * local_ce
            
            joint_optimizer.zero_grad()
            loss.backward()
            joint_optimizer.step()
            
            total_loss += loss.item()
            correct += (local_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        # Test
        local_model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                test_correct += (local_model(images).argmax(1) == labels).sum().item()
                test_total += labels.size(0)
        
        print(f"  Epoch {epoch+1}: loss={total_loss/len(trainloader):.4f} | Local Train {100*correct/total:.1f}% | Test {100*test_correct/test_total:.1f}%")
    
    # Save models
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(local_model.state_dict(), 'checkpoints/local_model.pt')
    torch.save(expert_model.state_dict(), 'checkpoints/expert_model.pt')
    torch.save(rejector.state_dict(), 'checkpoints/rejector.pt')
    
    # Evaluate
    print("\n=== Evaluation ===")
    evaluate(local_model, expert_model, rejector, testloader, device, args)


def train_classifier(model, trainloader, testloader, device, epochs=10, lr=0.001, name="Model"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in tqdm(trainloader, desc=f"{name} Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        # Test accuracy
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_correct += (outputs.argmax(1) == labels).sum().item()
                test_total += labels.size(0)
        
        print(f"  {name} Epoch {epoch+1}: Train {100*correct/total:.1f}% | Test {100*test_correct/test_total:.1f}%")


def evaluate(local_model, expert_model, rejector, testloader, device, args):
    """Evaluate system under varying number of clients."""
    local_model.eval()
    expert_model.eval()
    rejector.eval()
    
    post_hoc = PostHocRejector(abstain_cost=args.abstain_cost)
    
    results = {}
    
    for num_clients in [1, 2, 5, 10, 15, 20]:
        deferral_cost = simulate_deferral_cost(num_clients)
        
        # Method 1: Trained rejector
        trained_stats = eval_trained_rejector(local_model, expert_model, rejector,
                                              testloader, device)
        
        # Method 2: Post-hoc threshold adjustment
        posthoc_stats = eval_posthoc(local_model, expert_model, post_hoc,
                                      testloader, device, deferral_cost)
        
        # Baseline: Standard L2D (always defer if local uncertain, no reject option)
        baseline_stats = eval_baseline_l2d(local_model, expert_model,
                                            testloader, device, deferral_cost)
        
        results[num_clients] = {
            'deferral_cost': deferral_cost,
            'trained': trained_stats,
            'posthoc': posthoc_stats,
            'baseline': baseline_stats,
        }
        
        print(f"\n--- {num_clients} clients (c1={deferral_cost:.3f}) ---")
        print(f"  Baseline L2D:   acc={baseline_stats['accuracy']:.3f}  defer_rate={baseline_stats['defer_rate']:.3f}  avg_cost={baseline_stats['avg_cost']:.3f}")
        print(f"  Trained 3-logit: acc={trained_stats['accuracy']:.3f}  defer={trained_stats['defer_rate']:.3f}  reject={trained_stats['reject_rate']:.3f}  avg_cost={trained_stats['avg_cost']:.3f}")
        print(f"  Post-hoc:       acc={posthoc_stats['accuracy']:.3f}  defer={posthoc_stats['defer_rate']:.3f}  reject={posthoc_stats['reject_rate']:.3f}  avg_cost={posthoc_stats['avg_cost']:.3f}")
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_results(results)


def eval_trained_rejector(local_model, expert_model, rejector, testloader, device):
    correct, total, deferred, rejected = 0, 0, 0, 0
    total_cost = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            local_out = local_model(images)
            expert_out = expert_model(images)
            actions = rejector(images).argmax(dim=1)  # 0=accept, 1=defer, 2=reject
            
            for i in range(images.size(0)):
                if actions[i] == 0:  # Accept local
                    pred = local_out[i].argmax()
                    correct += (pred == labels[i]).item()
                elif actions[i] == 1:  # Defer to expert
                    pred = expert_out[i].argmax()
                    correct += (pred == labels[i]).item()
                    deferred += 1
                else:  # Reject
                    rejected += 1
                total += 1
    
    answered = total - rejected
    accuracy = correct / answered if answered > 0 else 0
    return {
        'accuracy': accuracy,
        'defer_rate': deferred / total,
        'reject_rate': rejected / total,
        'avg_cost': 1 - (correct / total),  # simplified
    }


def eval_posthoc(local_model, expert_model, post_hoc, testloader, device, deferral_cost):
    correct, total, deferred, rejected = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            local_probs = torch.softmax(local_model(images), dim=1)
            expert_probs = torch.softmax(expert_model(images), dim=1)
            actions = post_hoc.decide(local_probs, expert_probs, deferral_cost)
            
            for i in range(images.size(0)):
                if actions[i] == 0:
                    pred = local_probs[i].argmax()
                    correct += (pred == labels[i]).item()
                elif actions[i] == 1:
                    pred = expert_probs[i].argmax()
                    correct += (pred == labels[i]).item()
                    deferred += 1
                else:
                    rejected += 1
                total += 1
    
    answered = total - rejected
    accuracy = correct / answered if answered > 0 else 0
    return {
        'accuracy': accuracy,
        'defer_rate': deferred / total,
        'reject_rate': rejected / total,
        'avg_cost': 1 - (correct / total),
    }


def eval_baseline_l2d(local_model, expert_model, testloader, device, deferral_cost):
    """Standard 2-action L2D: accept or defer (no reject)."""
    correct, total, deferred = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            local_probs = torch.softmax(local_model(images), dim=1)
            expert_probs = torch.softmax(expert_model(images), dim=1)
            
            eta_m = local_probs.max(dim=1).values
            eta_e = expert_probs.max(dim=1).values
            
            for i in range(images.size(0)):
                # Standard L2D: defer if expert is better, else accept
                if eta_e[i] > eta_m[i]:
                    pred = expert_probs[i].argmax()
                    correct += (pred == labels[i]).item()
                    deferred += 1
                else:
                    pred = local_probs[i].argmax()
                    correct += (pred == labels[i]).item()
                total += 1
    
    return {
        'accuracy': correct / total,
        'defer_rate': deferred / total,
        'reject_rate': 0.0,
        'avg_cost': 1 - (correct / total) + deferral_cost * (deferred / total),
    }


def plot_results(results):
    clients = sorted(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(clients, [results[c]['baseline']['accuracy'] for c in clients], 'r-o', label='Baseline L2D')
    axes[0].plot(clients, [results[c]['trained']['accuracy'] for c in clients], 'b-s', label='Trained 3-logit')
    axes[0].plot(clients, [results[c]['posthoc']['accuracy'] for c in clients], 'g-^', label='Post-hoc')
    axes[0].set_xlabel('Number of Clients')
    axes[0].set_ylabel('Accuracy (on answered)')
    axes[0].set_title('Accuracy vs Clients')
    axes[0].legend()
    axes[0].grid(True)
    
    # Defer Rate
    axes[1].plot(clients, [results[c]['baseline']['defer_rate'] for c in clients], 'r-o', label='Baseline L2D')
    axes[1].plot(clients, [results[c]['trained']['defer_rate'] for c in clients], 'b-s', label='Trained 3-logit')
    axes[1].plot(clients, [results[c]['posthoc']['defer_rate'] for c in clients], 'g-^', label='Post-hoc')
    axes[1].set_xlabel('Number of Clients')
    axes[1].set_ylabel('Defer Rate')
    axes[1].set_title('Defer Rate vs Clients')
    axes[1].legend()
    axes[1].grid(True)
    
    # Total Cost
    axes[2].plot(clients, [results[c]['baseline']['avg_cost'] for c in clients], 'r-o', label='Baseline L2D')
    axes[2].plot(clients, [results[c]['trained']['avg_cost'] for c in clients], 'b-s', label='Trained 3-logit')
    axes[2].plot(clients, [results[c]['posthoc']['avg_cost'] for c in clients], 'g-^', label='Post-hoc')
    axes[2].set_xlabel('Number of Clients')
    axes[2].set_ylabel('Average Cost')
    axes[2].set_title('System Cost vs Clients')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("\nPlot saved to results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert-epochs', type=int, default=15)
    parser.add_argument('--joint-epochs', type=int, default=20)
    parser.add_argument('--ce-weight', type=float, default=0.5,
                        help='Weight for auxiliary CE loss to maintain local model classification')
    parser.add_argument('--abstain-cost', type=float, default=0.3)
    parser.add_argument('--max-clients', type=int, default=20)
    args = parser.parse_args()
    
    train_models(args)
