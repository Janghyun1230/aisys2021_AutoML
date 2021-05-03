from torchvision import transforms
import torch
import torchvision


def dataloader(batch_size=64, input_resolution=32, n_train=5000, n_valid=5000):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
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
    n_train = min(n_train, 50000 - n_valid)
    indices = torch.arange(n_train)
    small_trainset = torch.utils.data.Subset(trainset, indices)
    trainloader = torch.utils.data.DataLoader(small_trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)
    indices = torch.arange(50000 - n_valid, 50000)
    validset = torch.utils.data.Subset(trainset, indices)
    validloader = torch.utils.data.DataLoader(validset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0)

    print("\nDataloader is defined!")
    print(
        f"=> train {len(small_trainset)}, valid {len(validset)}, test {len(testset)}"
    )
    return trainloader, validloader, testloader