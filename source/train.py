import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import torchvision.transforms.functional as F
import timm
from tqdm import tqdm
import torch.nn.functional as F

from dino import DINO_MODEL
from dataloader import CustomDataset


class DINOLoss(nn.Module):
    ### CHATGPT generated function
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.register_buffer('center', torch.zeros(1, out_dim))
        
    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(2)  

        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # tea1, tea2

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: 
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))
        self.center = self.center * 0.9 + batch_center * 0.1


class Trainer:
    def __init__(self, student_model, teacher_model, optimizer, criterion, device):
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device

    def ema_update(self, teacher_model, student_model, momentum=0.996):
        for param_t, param_s in zip(teacher_model.parameters(), student_model.parameters()):
            if param_t.data.shape == param_s.data.shape:
                param_t.data = momentum * param_t.data + (1. - momentum) * param_s.data
        return teacher_model

    def train(self, teacher_dataloader, student_dataloader, num_epochs, accumulation_steps=1):
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            self.student_model.train()
            total_loss = 0
            batch_iter = zip(student_dataloader, teacher_dataloader)
            batch_iter = tqdm(batch_iter, 
                            total=min(len(student_dataloader), len(teacher_dataloader)), 
                            desc=f'Epoch {epoch+1}', 
                            leave=False)
            
            for i, (data_student, data_teacher) in enumerate(batch_iter):
                data_student = data_student[0].to(self.device)
                data_teacher = data_teacher[1].to(self.device)

                output_teacher = self.teacher_model(data_teacher)
                output_student = self.student_model(data_student)
                loss = self.criterion(output_student, output_teacher)
                
                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.ema_update(self.teacher_model, self.student_model)

                total_loss += loss.item() * accumulation_steps
                batch_iter.set_postfix({'loss': loss.item() * accumulation_steps})

            avg_loss = total_loss / len(batch_iter)
            print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

            if (epoch + 1) % 5 == 0:  
                save_path = f"checkpoints/dino_model_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'student_model_state_dict': self.student_model.state_dict(),
                    'teacher_model_state_dict': self.teacher_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, save_path)
                print(f"Checkpoint saved to {save_path}")


from torchvision.datasets import STL10

dataset = STL10(root='./data', split='unlabeled', download=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = DINOLoss(out_dim=1000).to(device)


batch_size = 2048  
accumulation_steps = 1  
num_epochs = 10


num_workers = 16  
prefetch_factor = 2

def dino_collate_function(batch): 
    crops = list(zip(*batch)) 
    return [torch.stack(crop) for crop in crops]

teacher_dataset = CustomDataset(dataset, transform_type='global')
teacher_data_loader = torch.utils.data.DataLoader(
    teacher_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor,
    persistent_workers=True, 
    collate_fn=dino_collate_function
)

student_dataset = CustomDataset(dataset, transform_type='local')
student_data_loader = torch.utils.data.DataLoader(
    student_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor,
    persistent_workers=True,  
    collate_fn=dino_collate_function
)

# Create models and optimizer with higher learning rate
teacher_model = DINO_MODEL(model_name="vit_base_patch16_224", img_size=224, out_dim=1000)
student_model = DINO_MODEL(model_name="vit_base_patch16_224", img_size=96, out_dim=1000)

# Move models to device first
teacher_model = teacher_model.to(device)
student_model = student_model.to(device)

# Optimizer with slightly higher learning rate and weight decay
optimizer = optim.AdamW(student_model.parameters(), lr=0.0005, weight_decay=0.04)

# Create trainer and start training
trainer = Trainer(student_model, teacher_model, optimizer, criterion, device)
trainer.train(teacher_data_loader, student_data_loader, num_epochs=num_epochs, accumulation_steps=accumulation_steps)



                

    



