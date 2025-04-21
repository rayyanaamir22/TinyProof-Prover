"""
Run a demo with Dummy implementations.
"""

# frameworks
from dummy_rmaxts import DummyRMaxTS

# utils
from dummy import DummyModel, DummyTokenizer, DummyVerifier


def example_A():
    """
    Basic linear sequence proof (narrow correct path)
    """

    # define tactics and mappings
    tactics = ["T1", "T2", "T3", "Wrong1", "Wrong2", "Wrong3", "Wrong4", "Wrong5"]
    tactic_to_id = {t: i for i, t in enumerate(tactics)}
    id_to_tactic = {i: t for t, i in tactic_to_id.items()}

    # define tactic dictionary for each state
    tactic_dict = {
        "Theorem: A": ["T1", "Wrong1"],
        "Theorem: A\nT1": ["T2", "Wrong2"],
        "Theorem: A\nT1\nT2": ["T3", "Wrong3"],
    }

    # identify the correct proof
    correct_proof = "Theorem: A\nT1\nT2\nT3"

    # initialize dummy objects
    tokenizer = DummyTokenizer(id_to_tactic)
    model = DummyModel(tactic_dict, tactic_to_id)
    verifier = DummyVerifier(correct_proof)

    # initialize RMaxTS with dummy components
    rmax_ts = DummyRMaxTS(model, tokenizer, verifier, num_beams=2, max_depth=3)

    # run the proof search
    input_thm = "Theorem: A"
    proof_steps = rmax_ts.generate_whole_proof(input_thm, iterations_per_sim=10)
    print("Search complete.")


def example_B():
    """
    Proof with multiple wrong options (wider search space)
    """
    tactics = ["Intro", "ApplyLemma", "WrongA", "WrongB", "WrongC", "WrongD"]
    tactic_to_id = {t: i for i, t in enumerate(tactics)}
    id_to_tactic = {i: t for t, i in tactic_to_id.items()}
    tactic_dict = {
        "Theorem: B": ["WrongA", "WrongB", "Intro"],
        "Theorem: B\nIntro": ["ApplyLemma", "WrongC", "WrongD"],
    }
    correct_proof = "Theorem: B\nIntro\nApplyLemma"

    tokenizer = DummyTokenizer(id_to_tactic)
    model = DummyModel(tactic_dict, tactic_to_id)
    verifier = DummyVerifier(correct_proof)
    rmax_ts = DummyRMaxTS(model, tokenizer, verifier, num_beams=3, max_depth=2)

    input_thm = "Theorem: B"
    proof_steps = rmax_ts.generate_whole_proof(input_thm, iterations_per_sim=10)
    print("Search complete.")


def example_C():
    """
    Proof with multiple paths to end goal, but only 1 is correct_proof.
    """
    tactics = ["Split", "Left", "Right", "Conclude", "Invalid1", "Invalid2", "Invalid3"]
    tactic_to_id = {t: i for i, t in enumerate(tactics)}
    id_to_tactic = {i: t for t, i in tactic_to_id.items()}
    tactic_dict = {
        "Theorem: C": ["Split", "Invalid1"],
        "Theorem: C\nSplit": ["Right", "Left", "Invalid2"],
        "Theorem: C\nSplit\nLeft": ["Conclude", "Invalid3"],
        "Theorem: C\nSplit\nRight": ["Conclude", "Invalid3"],
    }
    correct_proof = "Theorem: C\nSplit\nLeft\nConclude"

    tokenizer = DummyTokenizer(id_to_tactic)
    model = DummyModel(tactic_dict, tactic_to_id)
    verifier = DummyVerifier(correct_proof)
    rmax_ts = DummyRMaxTS(model, tokenizer, verifier, num_beams=3, max_depth=4)

    input_thm = "Theorem: C"
    proof_steps = rmax_ts.generate_whole_proof(input_thm, iterations_per_sim=15)
    print("Search complete.")

def example_with_lean():
    """
    Example with lean4 tactics: Proving symmetry of equality.
    """
    tactics = ["intro a b h", "rw [h]", "refl", "apply h", "simp", "exact h", "rw [Nat.add_comm]"]
    tactic_to_id = {t: i for i, t in enumerate(tactics)}
    id_to_tactic = {i: t for t, i in tactic_to_id.items()} 

    tactic_dict = {
        "Theorem: ∀ a b : Nat, a = b → b = a": ["intro a b h", "simp", "rw [Nat.add_comm]"],
        "Theorem: ∀ a b : Nat, a = b → b = a\nintro a b h": ["rw [h]", "apply h", "exact h"],
        "Theorem: ∀ a b : Nat, a = b → b = a\nintro a b h\nrw [h]": ["refl", "simp", "apply h"],
    }

    correct_proof = "Theorem: ∀ a b : Nat, a = b → b = a\nintro a b h\nrw [h]\nrefl"

    tokenizer = DummyTokenizer(id_to_tactic)
    model = DummyModel(tactic_dict, tactic_to_id)
    verifier = DummyVerifier(correct_proof)
    rmax_ts = DummyRMaxTS(model, tokenizer, verifier, num_beams=3, max_depth=3)

    input_thm = "Theorem: ∀ a b : Nat, a = b → b = a"
    proof_steps = rmax_ts.generate_whole_proof(input_thm, iterations_per_sim=10)
    print("Search complete.")

if __name__ == "__main__":
    example_with_lean()