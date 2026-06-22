
import os
import torch

from tqdm import tqdm
import torch.nn.functional as F

from generic import init_logger


# DEPRECATED! Use `optimize.py` instead!
class BaseTrainer:
    """
    A base trainer class for training the any generic model, which includes the training loop and validation loop, as well as logging and checkpointing.

    TODO: This could be made so darn great to train any kind of model with any kind of input that I need!
    TODO: Add options to save losses to file / dataframes, have model name, and other information.
    """
    def __init__(self, model, training_loader, validation_loader, optimizer, 
                 loss_function, device, precision='float32',
                 scheduler=None, num_epochs=10, save_dir=None):
        self.model = model
        self.device = device
        self.precision = precision
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.num_epochs = num_epochs
        self.project_dir = '../' + save_dir if save_dir is not None else f'../{PROJECT_DIR}'
        self.results_dir = f'../{self.project_dir}/results'
        self.models_dir = f'../{self.project_dir}/trained-models'
        self.data_dir = f'../data'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for databatch in tqdm(self.training_loader, desc=f"Training Epoch {epoch+1}/{self.num_epochs}"):
            inputs, targets = databatch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(self.training_loader.dataset)
        return avg_loss
    
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for databatch in tqdm(self.validation_loader, desc=f"Validating Epoch {epoch+1}/{self.num_epochs}"):
                inputs, targets = databatch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(self.validation_loader.dataset)
        return avg_loss

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save model checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(self.models_dir, f'calibrator_model_epoch{epoch+1}_valloss{val_loss:.6f}.pt')
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Saved new best model checkpoint to {checkpoint_path} with validation loss {val_loss:.6f}")
