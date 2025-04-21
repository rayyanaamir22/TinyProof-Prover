"""
Run a demo with Dummy implementations.
"""

# frameworks
from dummy_rmaxts import DummyRMaxTS

# utils
from dummy import DummyModel, DummyTokenizer, DummyVerifier


def example_with_dummies():
    """
    Launch an RMaxTS using the Dummy objects from dummy.py
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
    print("Generated proof:", proof_steps)


if __name__ == "__main__":
    example_with_dummies()