"""
MCTS Proof Search.

Missing optimizations:
- Lean Verifier
- Concurrent LLM Search Agents
"""

import math
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch

class StopOnNewline(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        if self.tokenizer.decode(input_ids[0][-1]) == '\n':
            return True
        return False

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # String: theorem + current proof tactics
        self.parent = parent
        self.children = []  # List of (tactic, prior_prob, child_node)
        self.visit_count = 0
        self.total_reward = 0.0

class MCTS:
    """
    MCTS for selecting next tactic via simulation.
    """
    def __init__(self, model, tokenizer, c=1.0, b=1.0, max_depth=10, num_beams=5):
        self.model = model
        self.tokenizer = tokenizer
        self.c = c  # Exploration constant for UCT
        self.b = b  # Intrinsic reward bonus (RMaxTS-inspired)
        self.max_depth = max_depth
        self.num_beams = num_beams
        self.root = None

    def search(self, theorem, num_iterations):
        """Run MCTS to find the best next tactic."""
        self.root = Node(theorem)
        for _ in range(num_iterations):
            node = self.select(self.root)
            if not self.is_terminal(node):
                self.expand(node)
                if node.children:
                    child = random.choice(node.children)[2]  # Randomly select a new child
                    reward = self.simulate(child.state)
                else:
                    reward = 0.0
            else:
                reward = 1.0 if self.is_proof_complete(node.state) else 0.0
            self.backpropagate(node, reward)
        if self.root.children:
            best_child = max(self.root.children, key=lambda x: x[2].total_reward / x[2].visit_count if x[2].visit_count > 0 else 0)
            return best_child[0]  # Return the best tactic
        return None

    def select(self, node):
        """Select the most promising node using UCT with exploration bonus."""
        while node.children:
            node = self.select_best_child(node)
        return node

    def select_best_child(self, node):
        """Choose the child with the highest UCT score."""
        total_visits = node.visit_count
        best_score = -float('inf')
        best_child = None
        for _, prior_prob, child in node.children:
            if child.visit_count == 0:
                score = float('inf')
            else:
                Q = child.total_reward / child.visit_count
                U = self.c * prior_prob * math.sqrt(math.log(total_visits) / child.visit_count)
                bonus = self.b / math.sqrt(child.visit_count)  # RMaxTS-inspired intrinsic reward
                score = Q + U + bonus
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, node):
        """Expand the node by generating new tactics with the LLM."""
        tactics = self.generate_tactics(node.state)
        for tactic, prior_prob in tactics:
            new_state = node.state + "\n" + tactic
            child_node = Node(new_state, parent=node)
            node.children.append((tactic, prior_prob, child_node))

    def simulate(self, state):
        """Simulate a proof path and assign a sparse reward."""
        current_state = state
        for _ in range(self.max_depth):
            tactics = self.generate_tactics(current_state, num_beams=1, do_sample=True)
            if not tactics:
                break
            tactic, _ = tactics[0]
            current_state += "\n" + tactic
        return 1.0 if random.random() < 0.01 else 0.0  # Simulate sparse rewards

    def backpropagate(self, node, reward):
        """Update visit counts and rewards up the tree."""
        while node:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def generate_tactics(self, state, num_beams=5, do_sample=False):
        """Generate possible next tactics using the LLM."""
        prompt = f"Theorem: {state}\nProof:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        stopping_criteria = StoppingCriteriaList([StopOnNewline(self.tokenizer)])
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams if not do_sample else 1,
            do_sample=do_sample,
            max_new_tokens=50,
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True,
            output_scores=True
        )
        tactics = []
        if do_sample:
            sequence = outputs.sequences[0]
            tactic = self.tokenizer.decode(sequence[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            tactics.append((tactic, 1.0))  # Prior not used in simulation
        else:
            sequences_scores = outputs.sequences_scores
            probs = torch.softmax(sequences_scores, dim=0)
            for i in range(num_beams):
                sequence = outputs.sequences[i]
                tactic = self.tokenizer.decode(sequence[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                prior_prob = probs[i].item()
                tactics.append((tactic, prior_prob))
        return tactics

    def is_terminal(self, node):
        """
        Return True if this node was legally reached (it is a valid mathematical formulation i.e. valid lean4 code)
        """
        raise NotImplementedError

    def is_proof_complete(self, state):
        """
        Return True if this state corresponds to a valid lean4 proof
        """
        raise NotImplementedError


if __name__ == "__main__":
    # load model
    model_name = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    mcts = MCTS(model, tokenizer)

    # input theorem
    theorem = "forall (a b : ℕ), a + b = b + a"
    best_tactic = mcts.search(theorem, num_iterations=100)
    print("Best next tactic:", best_tactic)