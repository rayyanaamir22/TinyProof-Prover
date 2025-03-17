"""
Monte-Carlo Tree Search objects.
"""

import math
import random

class MCTSNode:
    def __init__(self, state, ast=None, parent=None, move=None):
        """
        Represents a node in the proof search tree.
        
        Attributes:
            state (str): Lean4 proof state (e.g., the current state of the proof script).
            ast (dict): Parsed AST or structural data from Lean4.
            parent (MCTSNode): Parent node.
            move (str): The proof step (action) taken from the parent to reach this state.
            children (list): Child nodes.
            visits (int): Number of times this node was visited.
            value (float): Cumulative reward from rollouts.
            intrinsic_reward (float): Intrinsic reward for novelty in RMaxTS.
            is_terminal (bool): True if this node represents a terminal (complete) proof.
            untried_moves (list): List of moves yet to be expanded from this state.
        """
        self.state = state
        self.ast = ast  # Precise Lean4 structural data (parsed AST)
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.intrinsic_reward = 0.0
        self.is_terminal = False
        self.untried_moves = []  # Should be initialized with possible moves

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def update(self, reward):
        self.visits += 1
        self.value += reward

    def get_uct(self, exploration_constant=1.41):
        """
        Calculate the UCT (Upper Confidence Bound) value for this node.
        Nodes with zero visits return infinity to ensure exploration.
        """
        if self.visits == 0:
            return float("inf")
        # UCT formula with additional intrinsic reward term.
        return (self.value / self.visits) + exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        ) + self.intrinsic_reward

    def extract_state_info(self):
        """
        Returns a dictionary of this node's internal state for analysis.
        """
        return {
            "state": self.state,
            "ast": self.ast,
            "visits": self.visits,
            "value": self.value,
            "intrinsic_reward": self.intrinsic_reward,
            "move": self.move,
            "is_terminal": self.is_terminal,
            "num_children": len(self.children)
        }

class MCTS:
    def __init__(self, root, model_interface, verifier, rollout_depth=5, exploration_constant=1.41):
        """
        Monte-Carlo Tree Search with RMaxTS enhancements.
        
        Args:
            root (MCTSNode): The root node of the proof search tree.
            model_interface: An instance of the LLM interface for generating proof steps.
            verifier: An instance of the proof verifier (Lean4 interface).
            rollout_depth (int): Maximum depth for simulation rollouts.
            exploration_constant (float): Constant to balance exploration and exploitation.
        """
        self.root = root
        self.model_interface = model_interface
        self.verifier = verifier
        self.rollout_depth = rollout_depth
        self.exploration_constant = exploration_constant

    def select(self, node):
        """
        Traverse the tree from the given node using the UCT rule until reaching
        a node with untried moves or a terminal node.
        """
        while not node.is_terminal and not node.untried_moves and node.children:
            node = max(node.children, key=lambda child: child.get_uct(self.exploration_constant))
        return node

    def expand(self, node):
        """
        Expands the node by taking one untried move.
        Applies the move to generate a new state and corresponding AST.
        """
        if node.untried_moves:
            move = node.untried_moves.pop()
            # Apply move to get new Lean4 state; here we simply append the move.
            new_state = self.apply_move(node.state, move)
            new_ast = self.parse_ast(new_state)
            child_node = MCTSNode(state=new_state, ast=new_ast, parent=node, move=move)
            # Compute intrinsic reward based on state novelty.
            child_node.intrinsic_reward = self.compute_intrinsic_reward(child_node)
            # Check if the new state is terminal (dummy condition for illustration)
            if "end_proof" in new_state:
                child_node.is_terminal = True
            node.add_child(child_node)
            return child_node
        return node

    def simulate(self, node):
        """
        Perform a rollout (simulation) from the given node for a fixed depth.
        At each step, generate a move using the model and verify it.
        Returns the cumulative reward from the simulation.
        """
        current_state = node.state
        total_reward = 0.0
        for _ in range(self.rollout_depth):
            prompt = f"Continue proof from state:\n{current_state}\nProof step:"
            # Generate a move (proof step) from the model.
            generated_move = self.model_interface.infer(prompt, max_length=50)
            # Apply the generated move.
            new_state = self.apply_move(current_state, generated_move)
            # Verify the new state with the Lean4 proof assistant.
            result = self.verifier.verify_proof(new_state)
            if result.get("valid"):
                reward = 1.0  # Extrinsic reward for a valid proof step.
            else:
                reward = 0.0
            total_reward += reward
            current_state = new_state
            # Break early if we detect a terminal condition.
            if "end_proof" in new_state:
                break
        return total_reward

    def backpropagate(self, node, reward):
        """
        Backpropagate the reward up the tree.
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def apply_move(self, state, move):
        """
        Dummy state transformation: appends the move to the state.
        In a real implementation, this should apply Lean4-specific transformations.
        """
        return state + "\n" + move

    def parse_ast(self, state):
        """
        Dummy parser for Lean4 state into AST.
        In a real system, integrate the Lean4 AST parser here.
        """
        # For demonstration, we split the proof into steps.
        return {"steps": state.strip().splitlines()}

    def compute_intrinsic_reward(self, node):
        """
        Compute an intrinsic reward for the node.
        This reward incentivizes exploring novel proof states.
        """
        # Dummy computation: more reward if node is visited less.
        return 1.0 / (node.visits + 1)

    def run_search(self, iterations=100):
        """
        Execute the MCTS search for a given number of iterations.
        Returns the best node found (highest visit count).
        """
        for _ in range(iterations):
            # Selection: choose a node to expand.
            node = self.select(self.root)
            # Expansion: add a new child if possible.
            if not node.is_terminal:
                node = self.expand(node)
            # Simulation: rollout to get a reward.
            reward = self.simulate(node)
            # Backpropagation: update tree with the obtained reward.
            self.backpropagate(node, reward)
        # Choose the best child from the root for further processing.
        best_child = max(self.root.children, key=lambda n: n.visits, default=self.root)
        return best_child

# For testing and demonstration purposes:
if __name__ == "__main__":
    # Create a dummy root state.
    root_state = "start_proof"
    root_ast = {"steps": [root_state]}
    root_node = MCTSNode(state=root_state, ast=root_ast)
    # Initialize possible moves from the root.
    root_node.untried_moves = [
        "apply lemma1",
        "apply theorem2",
        "induction",
        "simplify"
    ]
    
    # Dummy model_interface that returns a fixed move.
    class DummyModelInterface:
        def infer(self, prompt, max_length=50, **kwargs):
            # For demonstration, return a dummy move.
            return "dummy_move"
    
    # Dummy verifier that marks proofs containing "dummy_move" as valid.
    class DummyVerifier:
        def verify_proof(self, proof, timeout=10):
            if "dummy_move" in proof:
                return {"valid": True, "errors": None, "ast": None}
            return {"valid": False, "errors": "dummy error", "ast": None}
    
    mcts_instance = MCTS(
        root=root_node,
        model_interface=DummyModelInterface(),
        verifier=DummyVerifier(),
        rollout_depth=3
    )
    
    best_node = mcts_instance.run_search(iterations=20)
    print("Best Node State Information:")
    print(best_node.extract_state_info())
