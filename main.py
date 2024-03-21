from bj import BlackjackGame
import numpy as np

###Convention: 1 is hit, 0 is stand

#Policy PI
pi = np.zeros((10, 10, 2), dtype=int)

#X axis is the player's sum
for i in range(12, 22):
    #Y axis is the dealer's card
    for j in range(1, 11):
        #Z axis is the usable ace
        for k in range(2):
            #If the player's sum is less than 20, the policy is to hit
            if i < 20:
                pi[i-12, j-1, k] = 1
            #If the player's sum is greater than 20, the policy is to stand
            else:
                pi[i-12, j-1, k] = 0

#Action value function Q for all states and actions
Q = np.zeros((10, 10, 2, 2))

#Returns list
returns = np.zeros((10, 10, 2, 2))

def get_random_state():
    player_sum = np.random.randint(0, 10)
    dealer_card = np.random.randint(0, 10)
    usable_ace = np.random.randint(2)
    return player_sum, dealer_card, usable_ace

def get_action(player_sum, dealer_card, usable_ace, pi):
    # Convert the values to indexes that match the policy structure
    player_sum_idx = player_sum - 12
    dealer_card_idx = dealer_card - 1
    usable_ace_idx = int(usable_ace)
    return pi[player_sum_idx, dealer_card_idx, usable_ace_idx]


if __name__ == "__main__":
    for _ in range(1):  # Number of episodes
        random_state = get_random_state()
        game = BlackjackGame(random_state)
        game.start_game()
        G = 0  # This will be the return (cumulative reward)
        episode = []

        while not game.game_over:
            current_state = game.get_game_state()
            player_sum = current_state["player_total"]
            dealer_showing = current_state["dealer_showing_card"]
            usable_ace = current_state["player_usable_ace"]
            action = get_action(player_sum, dealer_showing, usable_ace, pi)
            episode.append((player_sum, dealer_showing, usable_ace, action))  # Record the state-action pair

            # Take the action
            if action == 1:
                game.hit("player")
            else:
                game.stand()

        # Get reward for the episode
        reward = game.get_winner()
        # Apply the reward to all state-action pairs in the episode
        for state in episode:
            player_sum, dealer_showing, usable_ace, action = state
            # Update Q and pi based on the reward and the state-action pairs
            # This part of the code needs to be implemented
