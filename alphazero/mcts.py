"""
Monte Carlo Tree Search (MCTS) implementation for AlphaZero
"""

import numpy as np
import math


class MCTSNode:
    """Node in the MCTS tree"""

    def __init__(self, prior_prob):
        """
        Initialize MCTS node

        Args:
            prior_prob: Prior probability from neural network policy
        """
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob
        self.children = {}  # Maps action -> MCTSNode

    def value(self):
        """Get mean action value Q"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self):
        """Check if node is a leaf (no children)"""
        return len(self.children) == 0

    def select_child(self, c_puct):
        """
        Select child with highest UCB score

        Args:
            c_puct: Exploration constant

        Returns:
            tuple: (action, child_node)
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.get_ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def get_ucb_score(self, parent_visit_count, c_puct):
        """
        Calculate UCB score: Q + U
        U = c_puct * P * sqrt(parent_N) / (1 + N)

        Args:
            parent_visit_count: Visit count of parent node
            c_puct: Exploration constant

        Returns:
            float: UCB score
        """
        q_value = self.value()
        u_value = c_puct * self.prior_prob * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return q_value + u_value

    def expand(self, action_priors):
        """
        Expand node by creating children for all valid actions

        Args:
            action_priors: Dictionary mapping action -> prior probability
        """
        for action, prob in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(prob)

    def update(self, value):
        """
        Update node statistics

        Args:
            value: Value to back up (from current player's perspective)
        """
        self.visit_count += 1
        self.total_value += value


class MCTS:
    """Monte Carlo Tree Search"""

    def __init__(self, network, game, args):
        """
        Initialize MCTS

        Args:
            network: Neural network for policy and value prediction
            game: Game instance
            args: Configuration arguments (num_simulations, c_puct, etc.)
        """
        self.network = network
        self.game = game
        self.args = args

        self.root = None

    def get_action_probs(self, game, temperature=1.0, add_dirichlet_noise=False):
        """
        Get action probabilities from MCTS

        Args:
            game: Current game state
            temperature: Temperature for exploration (higher = more exploration)
            add_dirichlet_noise: Whether to add Dirichlet noise to root (for exploration)

        Returns:
            numpy.ndarray: Action probability distribution
        """
        # Run simulations
        for i in range(self.args['num_simulations']):
            # Clone game for simulation
            game_copy = game.clone()
            # Add Dirichlet noise only on first simulation when initializing root
            self._simulate(game_copy, add_dirichlet_noise=(add_dirichlet_noise and i == 0))

        # Get visit counts for all actions
        action_visits = np.zeros(game.get_action_size())
        for action, child in self.root.children.items():
            action_visits[action] = child.visit_count

        if temperature == 0:
            # Greedy: select most visited action
            action_probs = np.zeros(game.get_action_size())
            best_action = np.argmax(action_visits)
            action_probs[best_action] = 1.0
        else:
            # Sample proportionally to visit counts with temperature
            action_visits_temp = action_visits ** (1.0 / temperature)
            action_probs = action_visits_temp / np.sum(action_visits_temp)

        return action_probs

    def _simulate(self, game, add_dirichlet_noise=False):
        """
        Run one MCTS simulation

        Args:
            game: Game state to simulate from
            add_dirichlet_noise: Whether to add Dirichlet noise to root node

        Returns:
            float: Value from current player's perspective
        """
        # Check if game ended
        game_ended = game.get_game_ended()
        if game_ended != 0:
            # Game is over, return result
            return -game_ended  # Negative because we return from opponent's view

        # Initialize root if needed
        if self.root is None:
            policy, value = self.network.predict(game.get_canonical_board())
            valid_moves = game.get_valid_moves()
            policy = policy * valid_moves  # Mask invalid moves
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                # If all valid moves have zero probability, use uniform distribution
                policy = valid_moves / np.sum(valid_moves)

            # Add Dirichlet noise for exploration (during self-play)
            if add_dirichlet_noise:
                alpha = self.args.get('dirichlet_alpha', 0.3)
                epsilon = self.args.get('dirichlet_epsilon', 0.25)

                # Generate Dirichlet noise
                valid_actions = [a for a in range(len(policy)) if valid_moves[a] > 0]
                noise = np.random.dirichlet([alpha] * len(valid_actions))

                # Mix policy with noise: P_new = (1-ε)*P + ε*noise
                for i, action in enumerate(valid_actions):
                    policy[action] = (1 - epsilon) * policy[action] + epsilon * noise[i]

                # Renormalize
                policy_sum = np.sum(policy)
                if policy_sum > 0:
                    policy = policy / policy_sum

            action_priors = {a: policy[a] for a in range(len(policy)) if valid_moves[a] > 0}
            self.root = MCTSNode(0)
            self.root.expand(action_priors)
            return -value

        # Selection: traverse tree until leaf
        node = self.root
        search_path = [node]
        action_path = []

        while not node.is_leaf():
            action, node = node.select_child(self.args['c_puct'])
            search_path.append(node)
            action_path.append(action)
            game.get_next_state(action)

        # Check if game ended after selection
        game_ended = game.get_game_ended()
        if game_ended != 0:
            # Game is over
            value = -game_ended
        else:
            # Expansion and evaluation
            policy, value = self.network.predict(game.get_canonical_board())
            valid_moves = game.get_valid_moves()
            policy = policy * valid_moves
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                policy = valid_moves / np.sum(valid_moves)

            action_priors = {a: policy[a] for a in range(len(policy)) if valid_moves[a] > 0}
            node.expand(action_priors)

        # Backup: propagate value up the tree
        for node in reversed(search_path):
            node.update(value)
            value = -value  # Flip value for opponent

        return value

    def update_root(self, action):
        """
        Update root to the child corresponding to the action taken

        Args:
            action: Action that was taken
        """
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
            self.root.prior_prob = 0  # Root doesn't need prior
        else:
            self.root = None

    def reset(self):
        """Reset MCTS tree"""
        self.root = None


def get_mcts_policy(network, game, args, temperature=1.0):
    """
    Get MCTS policy for a given game state

    Args:
        network: Neural network
        game: Game state
        args: MCTS configuration
        temperature: Temperature for action selection

    Returns:
        numpy.ndarray: Action probability distribution
    """
    mcts = MCTS(network, game, args)
    action_probs = mcts.get_action_probs(game, temperature=temperature)
    return action_probs
