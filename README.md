# DINO from Scratch
![DINO Architecture](screenshots/attention_maps.png)


# Table of Contents
- [DINO from Scratch](#dino-from-scratch)
  - [Motivation](#motivation)
  - [Model Overview](#model-overview)
  - [Implementation Highlights](#implementation-highlights)
  - [Usage](#usage)
  - [Training & Monitoring](#training--monitoring)
  - [Key Hyperparameters](#key-hyperparameters)


## Motivation
This repository provides a from-scratch, research-oriented implementation of DINO (Self-Distillation with No Labels) for Vision Transformers (ViT). The goal is to offer a transparent, modular, and extensible codebase for:
- Experimenting with self-supervised learning (SSL) beyond the constraints of the original Facebook DINO repo
- Integrating DINO with custom datasets, backbones, or loss functions
- Benchmarking and ablation studies
- Gaining a deeper understanding of DINO's mechanisms and design

## Model Overview
DINO leverages self-distillation without labels, using a teacher-student paradigm:
- **Backbone:** Vision Transformer (ViT) via timm, with support for arbitrary patch sizes and input resolutions.
- **DINO Head:** A 5-layer MLP with GELU activations and LayerNorm, projecting the [CLS] token to a high-dimensional space (default: 1000).
- **Multi-crop strategy:** Each image yields multiple global (224x224) and local (96x96) crops. The teacher processes only global crops; the student processes both.
- **Teacher-Student:**
  - The student is trained by matching its output distribution to the teacher's for different views of the same image.
  - The teacher is updated as an exponential moving average (EMA) of the student.
- **DINO Loss:**
  - Cross-entropy between the softmaxed, temperature-scaled teacher and student outputs, with centering to prevent collapse.
  - Teacher outputs are sharpened (low temperature), student outputs are smoothed (higher temperature).
  - Centering is updated online to stabilize training.
- **No labels are used at any point.**

## Implementation Highlights
- **Full control over data pipeline:** Custom `CustomDataset` and collate for multi-crop, easily extensible to other datasets or crop strategies.
- **Backbone-agnostic:** Swap ViT for any timm-compatible model; patch size and input resolution are configurable per model.
- **Explicit device and memory management:** Designed for large-batch, multi-GPU training; supports gradient accumulation and efficient data loading.
- **Loss and EMA logic are modular:** Easy to adapt for other SSL paradigms (BYOL, MoCo, etc.) or to experiment with different centering/temperature schedules.
- **Minimal external dependencies:** Only PyTorch, timm, and tqdm; no reliance on Facebook's DINO codebase.

## Usage
- **Dataset:** Replace STL10 in `train.py` with any torchvision or custom dataset. The `CustomDataset` class expects a dataset returning PIL images.
- **Backbone:** Change `model_name` in `train.py` and `dino.py` to any timm model. Adjust `img_size` and `out_dim` as needed.
- **Augmentation:** Modify `get_global_transforms` and `get_local_transforms` in `dataloader.py` for custom multi-crop strategies.
- **Hyperparameters:** All key parameters (batch size, learning rate, temperatures, EMA momentum, etc.) are defined at the top of `train.py`.
- **Scaling:** Increase `batch_size` and `num_workers` to maximize GPU utilization. Use gradient accumulation for very large effective batch sizes.
- **Checkpointing:** Models and optimizer state are checkpointed every 5 epochs. Adjust frequency as needed.
- **Monitoring:** Training loss is tracked via tqdm. For more advanced logging, integrate with Weights & Biases or TensorBoard.
- **Integration:** The modular design allows for easy integration with other SSL methods or downstream tasks.

## Training & Monitoring
- Designed for high-throughput, large-batch training on modern GPUs (tested up to 46GB VRAM).
- Persistent DataLoader workers and pin_memory for efficient data transfer.
- Checkpoints include both student and teacher weights, optimizer, and loss.
- For distributed/multi-GPU, adapt the DataLoader and model wrapping as needed.

## Key Hyperparameters
- `batch_size`: 2048 (default; scale as memory allows)
- `num_workers`: 16
- `num_epochs`: 10
- `learning_rate`: 0.0005 (AdamW)
- `weight_decay`: 0.04
- `teacher_temp`: 0.04
- `student_temp`: 0.1
- `out_dim`: 1000
- `img_size`: 224 (teacher), 96 (student)
- `ema_momentum`: 0.996 (can be tuned)

## Limitations & Future Work
- **Computational Resources:** Training was performed on a single NVIDIA L40S GPU (48GB VRAM), taking ~15 minutes per epoch. This is significantly less compute than the original paper, which used 8 V100 16GB GPUs for multiple days.
- **Pre-trained Backbone:** To reduce computational requirements, this implementation uses a pre-trained ViT backbone instead of training from scratch like the original paper. While this affects the "true" self-supervised nature, it's a practical compromise for resource-constrained environments.
- **Future Improvements:**
  - Scale to multi-GPU training for larger batch sizes and faster convergence
  - Implement true from-scratch training of the ViT backbone
  - Add support for more advanced augmentation strategies
  - Integrate with modern training frameworks (DeepSpeed, FSDP)
  - Experiment with different architectures beyond ViT

## Learning Outcomes
Through this implementation, several key insights were gained:
- **Architecture Design:** Deep understanding of the teacher-student framework and how EMA updates maintain stability
- **Memory Management:** Practical experience with large-model training, gradient accumulation, and efficient data loading
- **Loss Dynamics:** Insights into how temperature scaling and centering prevent mode collapse in self-supervised learning
- **Resource Optimization:** Learned to make practical trade-offs (like using pre-trained backbones) while preserving core algorithmic insights
- **Distributed Training:** Exposure to the requirements and challenges of scaling to multi-GPU/multi-node setups, including data parallelism, synchronization, and communication overheads. Realized the importance of frameworks like PyTorch DDP, DeepSpeed, and FSDP for efficient scaling.
- **Mixed Precision Training:** Understanding the benefits and caveats of using mixed precision (FP16/BFloat16) to accelerate training and reduce memory usage, and how to integrate tools like `torch.cuda.amp`.
- **Reproducibility:** Gained appreciation for controlling random seeds, environment variables, and deterministic settings to ensure experiment reproducibility, especially in distributed settings.
- **Data Pipeline Bottlenecks:** Learned to profile and optimize the data pipeline (disk I/O, augmentation, prefetching, pinning) to keep the GPU fully utilized.
- **Scalability Mindset:** Adopted a mindset of designing for scalability and robustness from the start, anticipating the needs of future, larger experiments.


## References
- [DINO Paper (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [Original DINO Code (Facebook Research)](https://github.com/facebookresearch/dino)

## Demo

For a visual demonstration of DINO's emergent properties and self-supervised learning capabilities, see the original repo's video:

[![DINO Demo Video](https://img.youtube.com/vi/47885e80-ab68-11eb-9975-d61d5a919e13/0.jpg)](https://private-user-images.githubusercontent.com/46140458/116817761-47885e80-ab68-11eb-9975-d61d5a919e13.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDcxMDYyNTEsIm5iZiI6MTc0NzEwNTk1MSwicGF0aCI6Ii80NjE0MDQ1OC8xMTY4MTc3NjEtNDc4ODVlODAtYWI2OC0xMWViLTk5NzUtZDYxZDVhOTE5ZTEzLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTEzVDAzMTIzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTczZmExOWFiOWY0NWNjMTgzYzUwY2E5NDE4MWI4MDUyNDUzZGQwZTU3NTk0OWNmYmEwMTEyOWFiOWExODdmZjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Ge2NRHzO7cVe58xRRTgLMFrZQP5k8CU94LcEOM46Nac)

This video illustrates how DINO learns semantically meaningful features without supervision, enabling impressive clustering and attention visualization effects ([source](https://private-user-images.githubusercontent.com/46140458/116817761-47885e80-ab68-11eb-9975-d61d5a919e13.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDcxMDYyNTEsIm5iZiI6MTc0NzEwNTk1MSwicGF0aCI6Ii80NjE0MDQ1OC8xMTY4MTc3NjEtNDc4ODVlODAtYWI2OC0xMWViLTk5NzUtZDYxZDVhOTE5ZTEzLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTEzVDAzMTIzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTczZmExOWFiOWY0NWNjMTgzYzUwY2E5NDE4MWI4MDUyNDUzZGQwZTU3NTk0OWNmYmEwMTEyOWFiOWExODdmZjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Ge2NRHzO7cVe58xRRTgLMFrZQP5k8CU94LcEOM46Nac)).




