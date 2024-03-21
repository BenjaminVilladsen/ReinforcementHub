import numpy as np

# Constants
HIT = 0
STAND = 1
USABLE_ACE = [0, 1]  # 0 for no usable ace, 1 for usable ace
PLAYER_RANGE = range(12, 22)  # Player decision range
DEALER_RANGE = range(1, 11)  # Possible dealer showing cards
ACTIONS = [HIT, STAND]

# Rewards
LOSS = -1
WIN = 1
DRAW = 0

# Initialize value function V and policy arbitrarily
V = np.zeros((10, 10, 2))  # 10x10x2 for player, dealer, and ace states
policy = np.ones((10, 10, 2))  # Default policy to stand


def is_bust(hand_value):
    return hand_value > 21


def evaluate_hand(hand_value, usable_ace):
    # If hand value exceeds 21 and there's a usable ace, convert ace from 11 to 1
    if is_bust(hand_value) and usable_ace:
        return hand_value - 10
    return hand_value


def policy_evaluation(policy, V, theta=0.001, gamma=1.0):
    while True:
        delta = 0
        # Loop over all state spaces (player, dealer, usable_ace)
        for player in PLAYER_RANGE:
            for dealer in DEALER_RANGE:
                for ace in USABLE_ACE:
                    old_value = V[player - 12][dealer - 1][ace]
                    new_value = 0

                    # Evaluate the current policy
                    action = policy[player - 12][dealer - 1][ace]
                    if action == HIT:
                        # Compute expected value of hitting
                        for card in range(1, 11):
                            next_hand_value = evaluate_hand(player + card, ace)
                            if is_bust(next_hand_value):
                                # If bust, we lose
                                new_value += LOSS / 10.0
                            else:
                                # If not bust, continue with the new hand value
                                new_value += V[next_hand_value - 12][dealer - 1][ace] / 10.0
                    else:
                        # Compute expected value of standing
                        dealer_hand = dealer
                        while dealer_hand < 17:
                            # Dealer hits
                            for card in range(1, 11):
                                dealer_hand = evaluate_hand(dealer_hand + card, dealer_hand == 11)
                        # Compare hands to determine reward
                        if dealer_hand > 21 or player > dealer_hand:
                            new_value = WIN
                        elif player < dealer_hand:
                            new_value = LOSS
                        else:
                            new_value = DRAW

                    # Write back the value into the value function V
                    V[player - 12][dealer - 1][ace] = new_value
                    delta = max(delta, np.abs(old_value - new_value))
        if delta < theta:
            break


def policy_improvement(V, policy):
    policy_stable = True
    for player in PLAYER_RANGE:
        for dealer in DEALER_RANGE:
            for ace in USABLE_ACE:
                old_action = policy[player - 12][dealer - 1][ace]

                # New action is the one that maximizes the expected return
                action_values = np.zeros(len(ACTIONS))
                for action in ACTIONS:
                    if action == HIT:
                        # Compute expected value of hitting
                        for card in range(1, 11):
                            next_hand_value = evaluate_hand(player + card, ace)
                            if is_bust(next_hand_value):
                                # If bust, we lose
                                action_values[HIT] += LOSS / 10.0
                            else:
                                # If not bust, continue with the new hand value
                                action_values[HIT] += V[next_hand_value - 12][dealer - 1][ace] / 10.0
                    else:
                        # Compute expected value of standing
                        dealer_hand = dealer
                        while dealer_hand < 17:
                            # Dealer hits
                            for card in range(1, 11):
                                dealer_hand = evaluate_hand(dealer_hand + card, dealer_hand == 11)
                        # Compare hands to determine reward
                        if dealer_hand > 21 or player > dealer_hand:
                            action_values[STAND] = WIN
                        elif player < dealer_hand:
                            action_values[STAND] = LOSS
                        else:
                            action_values[STAND] = DRAW

                # Select the best action
                new_action = np.argmax(action_values)
                policy[player - 12][dealer - 1][ace] = new_action
                if new_action != old_action:
                    policy_stable = False
    return policy_stable


# Policy iteration loop
while True:
    policy_evaluation(policy, V)
    if policy_improvement(V, policy):
        break

# Display the policy
print("Optimal policy (0: Hit, 1: Stand):")
print(policy)