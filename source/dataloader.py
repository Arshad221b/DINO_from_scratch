import torch 
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import STL10

dataset = STL10(root='./data', split='unlabeled', download=True)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, views = 2, transform_type = 'global'):
        self.root = root
        self.transform_type = transform_type
        self.views = views
    def __len__(self):
        return len(self.root)

    def get_global_transforms(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_local_transforms(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.2, 1.)),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        index = index % len(self.root)
        image, _ = self.root[index]
        if self.transform_type == "global":  
            transforms = self.get_global_transforms()
            image =[transforms(image) for _ in range(self.views)]
        elif self.transform_type == "local":
            transforms = self.get_local_transforms()
            image = transforms(image)
        return image
    
def dino_collate_function(batch): 
    crops = list(zip(*batch)) 
    return [torch.stack(crop) for crop in crops]

train_dataset = CustomDataset(dataset, transform_type='global')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=dino_collate_function)

for i, data in enumerate(train_loader):
    print(data[0].shape, data[0][0].shape)
    break
