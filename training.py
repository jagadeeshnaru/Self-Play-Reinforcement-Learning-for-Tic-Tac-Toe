import random

from environment import TicTacToe
from agent import RLAgent
from minimax import MinimaxOpponent


def train_agent(episodes=40000):
    env = TicTacToe()
    agent = RLAgent()

    print(f"Training agent for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        done = False
        state_history = []
        final_reward = 0.0

        while not done:
            actions = env.available_actions()

            if not actions:
                break

            if env.current_player == 1:
                # Save the state before the agent chooses its move.
                state_history.append(state)
                action = agent.choose_action(env.board, actions)
            else:
                # During training, the opponent makes random legal moves.
                action = random.choice(actions)

            next_state, reward, done = env.step(action)
            state = next_state

            if done:
                final_reward = reward

        # Update the agent's decision states from the end of the game backward.
        for i in reversed(range(len(state_history))):
            current_state = state_history[i]

            if i == len(state_history) - 1:
                agent.update_value(current_state, final_reward, terminal=True)
            else:
                next_agent_state = state_history[i + 1]
                agent.update_value(current_state, 0.0, next_agent_state)

        # Gradually reduce random exploration as training progresses.
        if episode > 0 and episode % 5000 == 0:
            agent.epsilon = max(0.05, agent.epsilon * 0.9)

    print("Training complete!")
    return agent


def evaluate_agent(agent, num_games=1000):
    env = TicTacToe()
    minimax = MinimaxOpponent()

    wins = 0
    losses = 0
    draws = 0

    # Evaluation uses the learned policy without random exploration.
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    print(f"Evaluating agent against Minimax for {num_games} games...")

    for _ in range(num_games):
        env.reset()
        done = False

        while not done:
            actions = env.available_actions()

            if not actions:
                break

            if env.current_player == 1:
                action = agent.choose_action(env.board, actions)
            else:
                action = minimax.get_action(env.board)

            _, reward, done = env.step(action)

            if done:
                if reward == 1.0:
                    wins += 1
                elif reward == -1.0:
                    losses += 1
                else:
                    draws += 1

    agent.epsilon = old_epsilon

    print("\n--- Evaluation Results ---")
    print(f"Wins:   {wins} ({wins / num_games * 100:.2f}%)")
    print(f"Losses: {losses} ({losses / num_games * 100:.2f}%)")
    print(f"Draws:  {draws} ({draws / num_games * 100:.2f}%)")
