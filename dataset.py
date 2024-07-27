import torch
from torchvision import datasets, transforms



def get_dataset(data_path, num_samples=None):
    data_transform = transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_path, data_transform)

    if num_samples is not None:
        # sample random subset from train dataset
        subset_indexes = torch.randperm(len(dataset))[:num_samples]
        dataset = torch.utils.data.Subset(dataset, subset_indexes)
        
    # train_set = next(iter(torch.utils.data.DataLoader(
    #         dataset, batch_size=len(dataset), num_workers=4)))[0].cuda()    
    return dataset