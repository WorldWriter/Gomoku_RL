"""
AlphaZero Neural Network
ResNet architecture with policy and value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""

    def __init__(self, channels):
        """
        Args:
            channels: Number of channels
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Forward pass with residual connection"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out


class AlphaZeroNet(nn.Module):
    """
    AlphaZero neural network with ResNet backbone
    Outputs both policy and value predictions
    """

    def __init__(self, board_size, num_channels=128, num_res_blocks=10):
        """
        Initialize network

        Args:
            board_size: Size of the game board
            num_channels: Number of channels in residual blocks
            num_res_blocks: Number of residual blocks
        """
        super(AlphaZeroNet, self).__init__()

        self.board_size = board_size
        self.action_size = board_size * board_size

        # Input: 3 channels (current player, opponent, turn indicator)
        # Initial convolution block
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 3, board_size, board_size)

        Returns:
            tuple: (policy_logits, value)
                - policy_logits: (batch_size, action_size)
                - value: (batch_size, 1) in range [-1, 1]
        """
        # Initial convolution
        out = self.conv_input(x)
        out = self.bn_input(out)
        out = F.relu(out)

        # Residual tower
        for res_block in self.res_blocks:
            out = res_block(out)

        # Policy head
        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)  # (batch, action_size)

        # Value head
        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)  # Output in [-1, 1]

        return policy, value

    def predict(self, board_state):
        """
        Predict policy and value for a single board state

        Args:
            board_state: numpy array or tensor of shape (3, board_size, board_size)

        Returns:
            tuple: (policy, value)
                - policy: numpy array of shape (action_size,)
                - value: float
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(board_state, torch.Tensor):
                board_state = torch.FloatTensor(board_state)

            # Add batch dimension
            if board_state.dim() == 3:
                board_state = board_state.unsqueeze(0)

            board_state = board_state.to(next(self.parameters()).device)

            policy_logits, value = self.forward(board_state)

            # Convert to probabilities
            policy = F.softmax(policy_logits, dim=1)

            policy = policy.cpu().numpy()[0]
            value = value.cpu().numpy()[0][0]

        return policy, value

    def get_device(self):
        """Get the device the model is on"""
        return next(self.parameters()).device


def create_alphazero_net(board_size, num_channels=None, num_res_blocks=None):
    """
    Factory function to create AlphaZero network with size-appropriate defaults

    Args:
        board_size: Size of the board (5, 10, or 15)
        num_channels: Number of channels (default: auto-scaled by board size)
        num_res_blocks: Number of residual blocks (default: auto-scaled by board size)

    Returns:
        AlphaZeroNet: Neural network instance
    """
    # Default hyperparameters based on board size
    if num_channels is None:
        if board_size == 5:
            num_channels = 64
        elif board_size == 10:
            num_channels = 128
        else:  # 15
            num_channels = 256

    if num_res_blocks is None:
        if board_size == 5:
            num_res_blocks = 4
        elif board_size == 10:
            num_res_blocks = 8
        else:  # 15
            num_res_blocks = 10

    return AlphaZeroNet(board_size, num_channels, num_res_blocks)
