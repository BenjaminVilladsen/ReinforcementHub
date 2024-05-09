import numpy as np
import gym


def sarsa(epsilon_greedy_policy_fn, discretize_fn, q_table, env, settings):
    bins = [np.linspace(b[0], b[1], settings['num_bins']) for b in settings['state_bounds']]

    episode_rewards = []

    for episode in range(settings['num_episodes']):
        initial_state = env.reset()
        current_state = discretize_fn(initial_state, bins)
        current_action = epsilon_greedy_policy_fn(current_state, settings['epsilon'], env, q_table)
        done = False

        episode_reward = 0
        t = 0
        while not done:
            t += 1
            next_state_raw, reward, done, _, _ = env.step(current_action)  # Environment step
            episode_reward += reward
            next_state = discretize_fn(next_state_raw, bins)  # Discretize the resulting state
            next_action = epsilon_greedy_policy_fn(next_state, settings['epsilon'], env,
                                                   q_table)  # Choose next action using epsilon-greedy

            # SARSA update
            td_target = reward + settings['gamma'] * q_table[next_state + (next_action,)]
            td_delta = td_target - q_table[current_state + (current_action,)]
            q_table[current_state + (current_action,)] += settings['alpha'] * td_delta

            # Move to the next state and action
            current_state, current_action = next_state, next_action

        if episode % settings['log_interval'] == 0 and episode > 0:
            # print average reward the most recent 100 episodes
            print(f"Episode {episode}: Average reward: {np.mean(episode_rewards[-100:])}")
        episode_rewards.append(episode_reward)

    env.close()
    return q_table, episode_rewards
