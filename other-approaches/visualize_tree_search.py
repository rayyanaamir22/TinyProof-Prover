"""
Visualize a Tree Search with GraphViz
"""

# frameworks
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from graphviz import Digraph
import time

class TreeNode:
    def __init__(self, state, parent=None):
        self.state = state  # Textual state (e.g., proof step)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0  # Estimated value from DeepSeekProver

class SimplifiedRMaxTS:
    def __init__(self, model_name="deepseek-ai/DeepSeek-Prover-V1.5-RL"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tree = None
        self.visualize_tree = True
        self.graph = Digraph(comment="RMaxTS Search Tree")

    def evaluate_state(self, state):
        """Query DeepSeekProver to evaluate a state (proof step)."""
        inputs = self.tokenizer(state, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

    def expand(self, node, max_depth=3):
        """Expand the tree up to max_depth."""
        if max_depth == 0:
            return

        # Simulate possible next steps (in a real scenario, this would be dynamic)
        possible_actions = [
            f"{node.state} -> Step A",
            f"{node.state} -> Step B",
            f"{node.state} -> Step C",
        ]

        for action in possible_actions:
            child = TreeNode(action, parent=node)
            child.value = len(action) * 0.1  # Dummy value (replace with DeepSeek eval)
            node.children.append(child)
            self.expand(child, max_depth - 1)

    def search(self, initial_state, iterations=5):
        """Run RMaxTS from an initial state."""
        self.tree = TreeNode(initial_state)
        
        for i in range(iterations):
            print(f"\n--- Iteration {i+1} ---")
            self.expand(self.tree)
            
            # Update visualization
            if self.visualize_tree:
                self.visualize()
                time.sleep(1)  # Pause to see updates

    def visualize(self):
        """Update the Graphviz tree visualization."""
        self.graph = Digraph()
        self._add_nodes(self.tree)
        self.graph.render("rmax_tree", view=True, format="png")

    def _add_nodes(self, node):
        """Recursively add nodes to the graph."""
        self.graph.node(str(id(node)), f"{node.state}\nValue: {node.value:.2f}")
        for child in node.children:
            self.graph.edge(str(id(node)), str(id(child)))
            self._add_nodes(child)

if __name__ == "__main__":
    searcher = SimplifiedRMaxTS()
    initial_state = "Start Proof"
    searcher.search(initial_state)