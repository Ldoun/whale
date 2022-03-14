import torch

#target input: gathered ids list

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.round(output)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / (target.size(0) * target.size(1))

#cannot use top_k_acc for multi label problem
def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for j in range(k):
            correct += torch.sum(pred[:, j] == target).item()
    return correct / (target.size(0) * target.size(1))

