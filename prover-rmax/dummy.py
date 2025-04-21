"""
Dummy implementations to demonstrate how RMaxTS works.
"""

# frameworks
import random
import torch

class NotAVerifier:
    def __init__(self):
        pass

    def verify(self, state):
        return True, True

class DummyVerifier:
    def __init__(self, correct_proof):
        self.correct_proof = correct_proof.split("\n")

    def verify(self, state):
        state_lines = state.split("\n")
        if not state_lines[0].startswith("Theorem: "):
            return False, False
        for i, line in enumerate(state_lines[1:]):
            if i >= len(self.correct_proof) - 1 or line != self.correct_proof[i + 1]:
                return False, False
        if len(state_lines) == len(self.correct_proof):
            return True, True
        return True, False

class DummyTokenizer:
    def __init__(self, id_to_tactic):
        self.id_to_tactic = id_to_tactic
        self.eos_token_id = 999  # Dummy EOS token ID

    def __call__(self, text, return_tensors="pt"):
        inputs = {"input_ids": torch.tensor([[100]], dtype=torch.long)}
        return inputs

    def decode(self, ids, skip_special_tokens=True):
        # Convert tensor to list if needed
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # Ensure ids is a single ID
        if len(ids) == 1 and ids[0] in self.id_to_tactic:
            return self.id_to_tactic[ids[0]]
        print(f"Decode failed: ids={ids}, id_to_tactic={self.id_to_tactic}")  # Debug
        return "Unknown"

class DummyModel:
    def __init__(self, tactic_dict, tactic_to_id):
        self.tactic_dict = tactic_dict
        self.tactic_to_id = tactic_to_id
        self.current_state = None
        self.device = "cpu"

    def set_state(self, state):
        self.current_state = state

    def generate(self, input_ids, num_beams=1, do_sample=False, max_new_tokens=1, **kwargs):
        if self.current_state is None:
            raise ValueError("State not set")
        if self.current_state in self.tactic_dict:
            tactics = self.tactic_dict[self.current_state]
        else:
            tactics = ["Wrong4", "Wrong5"]
        if do_sample:
            tactic = random.choice(tactics)
            tactic_ids = [self.tactic_to_id[tactic]]
        else:
            tactic_ids = [self.tactic_to_id[t] for t in tactics[:num_beams]]
        # Return sequences with shape [num_return_sequences, sequence_length]
        sequences = [[100] + [tid] for tid in tactic_ids]
        return torch.tensor(sequences, dtype=torch.long, device=self.device)