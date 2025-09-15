import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from resnet_s import resnet20
from prepare_data import prepare_dataset
from grad_estim import apply_grad, gradient_estimation
from pruner import importance_score_grasp, gmask_lpr


def train_epoch(model, train_loader, criterion, optimizer, device, method="fo", prune_mask=None):
    model.train()

    num_processed = 0
    num_correct = 0
    sum_loss = 0.0

    pbar = tqdm(train_loader, total=len(train_loader), leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        grad_dict = gradient_estimation(model, method="fo", prune_mask=prune_mask)
        apply_grad(model, grad_dict)
        optimizer.step()

        sum_loss += loss.item()
        num_processed += data.size(0)
        _, predicted = output.max(1)
        num_correct += predicted.eq(target).sum().item()

        avg_loss = sum_loss / (batch_idx + 1)
        avg_acc = 100. * num_correct / num_processed

        pbar.set_description(f"T LOSS: {avg_loss:.4f}, T ACC: {avg_acc:.2f}%")

    return avg_loss, avg_acc

@torch.no_grad()
def test_epoch(model, test_loader, criterion, device):
    model.eval()

    num_processed = 0
    num_correct = 0
    sum_loss = 0.0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        sum_loss += criterion(output, target).item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        num_correct += pred.eq(target.view_as(pred)).sum().item()
        num_processed += data.size(0)

    avg_loss = sum_loss / len(test_loader)
    accuracy = 100. * num_correct / num_processed
    return avg_loss, accuracy

def main(args):
    num_epoch = args.epoch

    model = resnet20()
    num_params = sum(p.numel() for p in model.parameters())

    data_loaders, cls_num = prepare_dataset("cifar10", batch_size=args.batch_size)
    train_loader = data_loaders['train']
    test_loader = data_loaders['test']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion.to(device)

    # Pruning
    importance_score = importance_score_grasp(model, train_loader, criterion)
    prune_mask = gmask_lpr(model, importance_score, prune_ratio=args.prune_ratio)

    num_alive = sum(pmask.sum().item() for pname, pmask in prune_mask.items())
    print(f"Number of remaining parameters: {num_alive/1e3:.2f}K/{num_params/1e3:.2f}K ({num_alive/num_params*100:.2f}%)")

    # Training loop
    top_acc = 0.0
    for epoch in range(1, 1+num_epoch):
        if epoch <= 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * epoch / 3
        lr_epoch = param_group['lr']

        prune_mask = gmask_lpr(model, importance_score, prune_ratio=0.9)
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device, method=args.method, prune_mask=prune_mask)
        v_loss, v_acc = test_epoch(model, test_loader, criterion, device)

        is_best = v_acc > top_acc
        if is_best:
            top_acc = v_acc

        print(f"Epoch {epoch:3d}, LR {lr_epoch:.4e} | T LOSS: {t_loss:.4f}, T ACC: {t_acc:.2f}%, V LOSS: {v_loss:.4f}, V ACC: {v_acc:.2f}% | " + ("*" if is_best else ""))
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--method', type=str, choices=['fo', 'rge', 'cge'], default='fo', help='gradient estimation method (default: fo)')
    parser.add_argument('--prune_ratio', type=float, default=0.9, help='pruning ratio (default: 0.9)')
    
    args = parser.parse_args()
    print(args)
    
    main(args)
