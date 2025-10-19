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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
from datetime import datetime

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
        # 添加.copy()以解决负步长问题
        return (
            torch.FloatTensor(state.copy()),
            torch.FloatTensor(policy.copy() if isinstance(policy, np.ndarray) else policy),
            torch.FloatTensor([value])
        )


class Trainer:
    """
    AlphaZero trainer
    Manages the complete training pipeline
    """

    def __init__(self, network, game, args, device, log_dir='logs'):
        """
        Initialize trainer

        Args:
            network: Neural network
            game: Game instance
            args: Training configuration
            device: torch device (CPU/CUDA)
            log_dir: Directory for logs and visualizations
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

        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training start time
        self.start_time = datetime.now()
        self.training_log = []

    def train(self, num_iterations):
        """
        Main training loop

        Args:
            num_iterations: Number of training iterations
        """
        for iteration in range(1, num_iterations + 1):
            iteration_start = datetime.now()

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

            # Record training log
            iteration_time = (datetime.now() - iteration_start).total_seconds()
            log_entry = {
                'iteration': iteration,
                'loss': avg_loss,
                'buffer_size': len(self.train_examples_history),
                'time_seconds': iteration_time,
                'timestamp': datetime.now().isoformat()
            }
            self.training_log.append(log_entry)

            # Save checkpoint and logs
            print("\n[3/3] Saving checkpoint and logs...")
            self._save_checkpoint(iteration)
            self._save_training_log()

            # Plot training curve every 10 iterations or at the end
            if iteration % 10 == 0 or iteration == num_iterations:
                self._plot_training_curve(iteration)

            print(f"\nIteration {iteration} completed. Avg Loss: {avg_loss:.4f} | Time: {iteration_time:.1f}s")

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
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'args': self.args,
        }

        # Always save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        print(f"Latest checkpoint saved: {latest_path}")

        # Save numbered checkpoint at intervals
        save_interval = self.args.get('save_interval', 100)
        if iteration % save_interval == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Milestone checkpoint saved: {checkpoint_path}")

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

    def _save_training_log(self):
        """Save training log to JSON file"""
        log_path = self.log_dir / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump({
                'start_time': self.start_time.isoformat(),
                'config': self.args,
                'training_log': self.training_log,
                'loss_history': self.loss_history
            }, f, indent=2)

    def _plot_training_curve(self, current_iteration):
        """
        Plot and save training curve

        Args:
            current_iteration: Current training iteration
        """
        if not self.loss_history:
            return

        plt.figure(figsize=(14, 5))

        # Plot 1: Loss curve
        plt.subplot(1, 2, 1)
        iterations = list(range(1, len(self.loss_history) + 1))
        plt.plot(iterations, self.loss_history, 'b-', linewidth=1.5, alpha=0.7, label='Loss')

        # Add smoothed curve
        if len(self.loss_history) >= 10:
            window = min(20, len(self.loss_history) // 5)
            smoothed = []
            for i in range(len(self.loss_history)):
                start = max(0, i - window + 1)
                smoothed.append(sum(self.loss_history[start:i+1]) / (i - start + 1))
            plt.plot(iterations, smoothed, 'r-', linewidth=2, label=f'Smoothed (MA-{window})')

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics box
        stats_text = f"Current: {self.loss_history[-1]:.4f}\n"
        stats_text += f"Min: {min(self.loss_history):.4f}\n"
        stats_text += f"Max: {max(self.loss_history):.4f}"
        plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=10)

        # Plot 2: Loss reduction
        plt.subplot(1, 2, 2)
        if len(self.loss_history) > 1:
            initial_loss = self.loss_history[0]
            reduction = [(initial_loss - loss) / initial_loss * 100 for loss in self.loss_history]
            plt.plot(iterations, reduction, 'g-', linewidth=2)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss Reduction (%)', fontsize=12)
            plt.title('Loss Reduction from Initial', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Overall title
        board_size = self.args.get('board_size', 'N/A')
        total_games = current_iteration * self.args.get('num_self_play_games', 0)
        elapsed = (datetime.now() - self.start_time).total_seconds() / 3600

        plt.suptitle(f'AlphaZero Training Progress - {board_size}x{board_size} Board\n'
                     f'Iteration: {current_iteration} | Total Games: {total_games} | Time: {elapsed:.1f}h',
                     fontsize=16, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])

        # Save plot
        plot_path = self.log_dir / f'training_curve_iter_{current_iteration}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Also save as latest
        latest_plot = self.log_dir / 'training_curve_latest.png'
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(iterations, self.loss_history, 'b-', linewidth=1.5, alpha=0.7, label='Loss')
        if len(self.loss_history) >= 10:
            window = min(20, len(self.loss_history) // 5)
            smoothed = []
            for i in range(len(self.loss_history)):
                start = max(0, i - window + 1)
                smoothed.append(sum(self.loss_history[start:i+1]) / (i - start + 1))
            plt.plot(iterations, smoothed, 'r-', linewidth=2, label=f'Smoothed (MA-{window})')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        stats_text = f"Current: {self.loss_history[-1]:.4f}\nMin: {min(self.loss_history):.4f}\nMax: {max(self.loss_history):.4f}"
        plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

        plt.subplot(1, 2, 2)
        if len(self.loss_history) > 1:
            initial_loss = self.loss_history[0]
            reduction = [(initial_loss - loss) / initial_loss * 100 for loss in self.loss_history]
            plt.plot(iterations, reduction, 'g-', linewidth=2)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss Reduction (%)', fontsize=12)
            plt.title('Loss Reduction from Initial', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        plt.suptitle(f'AlphaZero Training Progress - {board_size}x{board_size} Board\n'
                     f'Iteration: {current_iteration} | Total Games: {total_games} | Time: {elapsed:.1f}h',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(latest_plot, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Training curve saved: {plot_path}")
