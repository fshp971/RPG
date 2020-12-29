import torchvision


def CIFAR10(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR10(root=root, train=train,
                        transform=transform, download=True)

def CIFAR100(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR100(root=root, train=train,
                        transform=transform, download=True)
