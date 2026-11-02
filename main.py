from training import train_agent, evaluate_agent


def main():
    # Train the RL agent through 40,000 games against a random opponent.
    trained_agent = train_agent(episodes=40000)

    # Test the trained agent against the Minimax opponent.
    evaluate_agent(trained_agent, num_games=1000)


if __name__ == "__main__":
    main()
