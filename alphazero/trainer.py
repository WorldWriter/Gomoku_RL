"""
AlphaZero Trainer
Main training loop integrating self-play, neural network training, and evaluation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from collections import deque

from .self_play import SelfPlay, augment_data


class TrainingDataset(Dataset):
    """Dataset for AlphaZero training"""

    def __init__(self, examples):
        """
        Args:
            examples: List of [state, policy, value]
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(policy),
            torch.FloatTensor([value])
        )


class Trainer:
    """
    AlphaZero trainer
    Manages the complete training pipeline
    """

    def __init__(self, network, game, args, device):
        """
        Initialize trainer

        Args:
            network: Neural network
            game: Game instance
            args: Training configuration
            device: torch device (CPU/CUDA)
        """
        self.network = network.to(device)
        self.game = game
        self.args = args
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=args.get('learning_rate', 0.001),
            weight_decay=args.get('weight_decay', 1e-4)
        )

        # Training history
        self.train_examples_history = deque(maxlen=args.get('max_history_len', 200000))
        self.loss_history = []

        # Create checkpoint directory
        self.checkpoint_dir = Path(args.get('checkpoint_dir', 'models'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, num_iterations):
        """
        Main training loop

        Args:
            num_iterations: Number of training iterations
        """
        for iteration in range(1, num_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{num_iterations}")
            print(f"{'='*60}")

            # Self-play
            print("\n[1/3] Generating self-play games...")
            train_examples = self._generate_self_play_data()

            # Add to history
            self.train_examples_history.extend(train_examples)
            print(f"Training buffer size: {len(self.train_examples_history)}")

            # Train neural network
            print("\n[2/3] Training neural network...")
            avg_loss = self._train_network()
            self.loss_history.append(avg_loss)

            # Save checkpoint
            print("\n[3/3] Saving checkpoint...")
            self._save_checkpoint(iteration)

            print(f"\nIteration {iteration} completed. Avg Loss: {avg_loss:.4f}")

    def _generate_self_play_data(self):
        """
        Generate training data through self-play

        Returns:
            list: Training examples
        """
        self.network.eval()

        num_games = self.args.get('num_self_play_games', 100)
        print(f"Playing {num_games} self-play games...")

        self_play = SelfPlay(self.network, self.game, self.args)
        train_examples = self_play.generate_games(num_games)

        # Data augmentation with symmetries
        if self.args.get('use_augmentation', True):
            print("Applying data augmentation...")
            train_examples = augment_data(train_examples, self.game)
            print(f"After augmentation: {len(train_examples)} examples")

        return train_examples

    def _train_network(self):
        """
        Train the neural network on collected examples

        Returns:
            float: Average loss
        """
        self.network.train()

        # Create dataset and dataloader
        dataset = TrainingDataset(list(self.train_examples_history))
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.get('batch_size', 64),
            shuffle=True,
            num_workers=0  # Use 0 for Windows compatibility
        )

        num_epochs = self.args.get('num_epochs', 10)
        total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_batches = 0

            for states, target_policies, target_values in dataloader:
                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)

                # Forward pass
                pred_policies, pred_values = self.network(states)

                # Calculate losses
                policy_loss = self._policy_loss(pred_policies, target_policies)
                value_loss = self._value_loss(pred_values, target_values)

                # Total loss
                loss = policy_loss + value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

            avg_epoch_loss = epoch_loss / epoch_batches
            print(f"  Epoch {epoch + 1}/{num_epochs}: Loss = {avg_epoch_loss:.4f}")

            total_loss += epoch_loss
            num_batches += epoch_batches

        avg_loss = total_loss / num_batches
        return avg_loss

    def _policy_loss(self, pred_policies, target_policies):
        """
        Policy loss (cross-entropy)

        Args:
            pred_policies: Predicted policy logits
            target_policies: Target policy probabilities

        Returns:
            torch.Tensor: Policy loss
        """
        # Cross-entropy between target and predicted policies
        return -torch.sum(target_policies * torch.log_softmax(pred_policies, dim=1)) / pred_policies.size(0)

    def _value_loss(self, pred_values, target_values):
        """
        Value loss (MSE)

        Args:
            pred_values: Predicted values
            target_values: Target values

        Returns:
            torch.Tensor: Value loss
        """
        return torch.mean((pred_values - target_values) ** 2)

    def _save_checkpoint(self, iteration):
        """
        Save model checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pth"

        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'args': self.args,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Also save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            int: Iteration number of loaded checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint.get('loss_history', [])

        iteration = checkpoint.get('iteration', 0)
        print(f"Resumed from iteration {iteration}")

        return iteration
