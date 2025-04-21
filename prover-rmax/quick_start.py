"""
Run a demo with an LLM.
"""

# frameworks
from transformers import AutoModelForCausalLM, AutoTokenizer

# utils
from dummy import NotAVerifier
from rmaxts import RMaxTS


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

if __name__ == '__main__':
    example_with_llm()