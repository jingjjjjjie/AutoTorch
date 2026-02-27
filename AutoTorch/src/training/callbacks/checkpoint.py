import os
import torch

class Checkpoint:
    """Saves model and optimizer state at the end of each epoch."""

    def __init__(self, save_dir, save_name):
        self.ckpt_dir = os.path.join(save_dir, save_name.replace('.pth', ''), 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, val_loss):
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        path = os.path.join(self.ckpt_dir, f'epoch_{epoch+1}.pt')
        torch.save(state, path)
