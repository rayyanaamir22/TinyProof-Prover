"""
Run some examples for the demo.
"""

# frameworks
from transformers import AutoModelForCausalLM, AutoTokenizer
from rmaxts import RMaxTS

# dummy implementations for demo
from dummy import DummyModel, DummyTokenizer, DummyVerifier, NotAVerifier


def example_with_llm():
    """
    Launch an LLM-directed RMaxTS.
    """

    # load model
    model_name = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # load verifier object
    # TODO: integrate DeepSeekProver's AST parser-based verifier
    verifier = NotAVerifier()  # this verifier deems any proof state (lean4 string) to be valid

    # problem setup
    rmax_ts = RMaxTS(model, tokenizer, verifier)
    theorem = "forall (a b : ℕ), a + b = b + a"
    
    # just get the next tactic
    best_tactic = rmax_ts.search_best_tactic(theorem, num_iterations=100)
    print("Best next tactic:", best_tactic)

    # search for an entire proof
    rmax_ts.generate_whole_proof(rmax_ts, theorem)


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
    rmax_ts = RMaxTS(model, tokenizer, verifier, num_beams=2, max_depth=3)

    # run the proof search
    proof_steps = rmax_ts.generate_whole_proof("Theorem: A", iterations_per_sim=10)
    print("Generated proof:", proof_steps)


if __name__ == "__main__":
    example_with_dummies()