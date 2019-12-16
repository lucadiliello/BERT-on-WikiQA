import numpy
import torch
import torch.nn as nn

def AP(output, target):
    _, indexes = torch.sort(output, descending=True)
    target = target[indexes].round()
    total = 0.
    for i in range(len(output)):
        index = i+1
        if target[i]:
            total += target[:index].sum().item() / index
    return total/target.sum().item()

def MAP(outputs, targets):
    assert(len(outputs) == len(targets))
    res = []
    for i in range(len(outputs)):
        res.append(AP(outputs[i], targets[i]))
    return numpy.mean(res)

def RR(output, target):
    _, indexes = torch.sort(output, descending=True)
    best = target[indexes].nonzero().squeeze().min().item()
    return 1.0/(best+1)

def MRR(outputs, targets):
    assert(len(outputs) == len(targets))
    res = []
    for i in range(len(outputs)):
        res.append(RR(outputs[i], targets[i]))
    return numpy.mean(res)