import torch

def train(model, device, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(1, epochs+1):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def test(model, device, criterion, test_loader):
	global client_id
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += criterion(output, target)
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss /= len(test_loader.dataset)
	acc = correct / len(test_loader.dataset)
	return test_loss, acc
