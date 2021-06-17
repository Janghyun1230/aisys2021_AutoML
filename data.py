from torchvision import transforms
import torch
import torchvision


def dataloader(batch_size=64,
               input_resolution=32,
               n_train=50000,
               n_valid=5000,
               _print=True,
               download=False):
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

    indices = torch.arange(n_train)
    trainset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True,
                                             download=download,
                                             transform=train_transform)
    small_trainset = torch.utils.data.Subset(trainset, indices)
    trainloader = torch.utils.data.DataLoader(small_trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)

    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=download,
                                            transform=test_transform)
    indices = torch.arange(n_valid)
    validset = torch.utils.data.Subset(testset, indices)
    validloader = torch.utils.data.DataLoader(validset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

    indices = torch.arange(n_valid, len(testset))
    testsubset = torch.utils.data.Subset(testset, indices)
    testloader = torch.utils.data.DataLoader(testsubset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)

    if _print:
        print("\nDataloader is defined!")
        print(f"=> train {len(trainset)}, valid {len(validset)}, test {len(testsubset)}")
    return trainloader, validloader, testloader
