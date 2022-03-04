import torch

#target input: gathered ids list

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(torch.stack([t[pred[i]] == 1 for i,t in enumerate(target)]))
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for j in range(k):
            correct += torch.sum(torch.stack([t[pred[:,j,i]] == 1 for i,t in enumerate(target)]))
    return correct / len(target)

