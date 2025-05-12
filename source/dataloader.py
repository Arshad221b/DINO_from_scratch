import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

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
            transform = self.get_global_transforms()
        elif self.transform_type == "local":
            transform = self.get_local_transforms()
        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")
        
        

        image = torch.stack([transform(image) for _ in range(self.views)], dim=0)
        return image
        
    def ema_update(self, teacher_model, student_model, momentum=0.996):
        for param_t, param_s in zip(teacher_model.parameters(), student_model.parameters()):
            if param_t.data.shape == param_s.data.shape:
                param_t.data = momentum * param_t.data + (1. - momentum) * param_s.data
        return teacher_model

