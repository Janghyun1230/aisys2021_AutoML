import torch


def accuracy(preds, target, topk=5):
    N = target.shape[0]
    acc = 0
    for i, pred in enumerate(preds):
        for idx in torch.topk(pred, k=topk).indices.squeeze(0).tolist():
            if idx == target[i]:
                acc += 1
                break
    return acc / N


def train(model, optimizer, dataloader, device, loss_fn, eval_fn):
    model.train()
    avg_loss = 0
    avg_accuracy = 0
    if device == "cuda":
        model.cuda()

    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)

        preds = logits.softmax(dim=1)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        avg_accuracy += eval_fn(preds, targets)
        print(step, end='\r')
    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy


def test(model, dataloader, device, loss_fn, eval_fn):
    model.eval()
    avg_loss = 0
    avg_accuracy = 0
    if device == "cuda":
        model.cuda()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.softmax(dim=1)
            loss = loss_fn(preds, targets)
            avg_loss += loss.item()
            avg_accuracy += eval_fn(preds, targets)

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy
