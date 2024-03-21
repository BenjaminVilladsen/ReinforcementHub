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

def get_action(state):
    return pi[state]



if __name__ == "__main__":
    for _ in range(1):
        random_state = get_random_state()
        random_state = (8, 1, 0)
        print(random_state)
        game = BlackjackGame(random_state)
        G = 0
        while not game.game_over:
            action = get_action(random_state)
            if action == 1:
                game.hit("player")
            else:
                game.stand()
        print(game.player_hand)
        print(game.dealer_hand)
        print(game.get_game_state())

