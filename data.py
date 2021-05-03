from torchvision import transforms
import torch
import torchvision


def dataloader(batch_size=64, input_resolution=32, n_train=5000, n_valid=5000):
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(input_resolution),
        transforms.RandomCrop(input_resolution, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(input_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True,
                                             download=True,
                                             transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)
    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=test_transform)
    indices = torch.arange(n_valid)
    validset = torch.utils.data.Subset(trainset, indices)
    validloader = torch.utils.data.DataLoader(validset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)
    indices = torch.arange(n_valid, len(testset))
    testsubset = torch.utils.data.Subset(trainset, indices)
    testloader = torch.utils.data.DataLoader(testsubset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)

    print("\nDataloader is defined!")
    print(
        f"=> train {len(small_trainset)}, valid {len(validset)}, test {len(testset)}"
    )
    return trainloader, validloader, testloader