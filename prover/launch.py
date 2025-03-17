"""
Run the model on an exemplar theorem.
"""

# frameworks
import logging
import concurrent.futures
import time
from model_interface import ModelInterface
from verifier import ProofVerifier
from rl_agent import RLAgent

logging.basicConfig(level=logging.INFO)

def run_agent_episode(agent):
    best_node, episode_reward = agent.run_episode()
    return best_node.extract_state_info(), episode_reward

def main():
    # Initialize the components.
    model_interface = ModelInterface(model_name="gpt2")
    verifier = ProofVerifier(server_url="http://localhost:8000")
    theorem = "theorem: for all natural numbers n, sum_{i=1}^n i = n(n+1)/2"
    agent = RLAgent(
        model_interface=model_interface,
        verifier=verifier,
        theorem=theorem,
        rollout_depth=5,
        mcts_iterations=20
    )
    
    num_episodes = 5  # Number of concurrent RL episodes.
    
    # Use a ThreadPoolExecutor for concurrent episodes.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_episodes) as executor:
        futures = [executor.submit(run_agent_episode, agent) for _ in range(num_episodes)]
        for future in concurrent.futures.as_completed(futures):
            state_info, reward = future.result()
            print("RL Episode Result:")
            print(state_info)
            print("Reward:", reward)
            print("------")
            time.sleep(1)  # Simulate processing delay.
    
if __name__ == "__main__":
    main()
