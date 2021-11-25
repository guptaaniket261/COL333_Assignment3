import A3
import time
import matplotlib.pyplot as plt


mdp = A3.Taxi_MDP()
# mdp.get_rand_start()
policy_obj = A3.Policy()
alpha = 0.25
discount = 0.99
epsilon = 1e-18
batch_size = 10
nums_episodes = [2000, 2500, 3000,3500,4000,4500]

#### Q Learning ####
print("Q Learning")
start = time.time()

discounted_rewards = []
for num_episodes in nums_episodes:
    averaged_sum = 0
    for i in range(batch_size):
        policy = {state : 0 for state in mdp.states}
        learned_policy, discounted_reward = policy_obj.q_learning(mdp, policy, alpha, discount,epsilon, num_episodes = num_episodes)
        averaged_sum += discounted_reward
    averaged_sum /= batch_size
    discounted_rewards.append(discounted_reward)
plt.title("Discounted reward sum vs no. of episodes (Queue Learning)" )
plt.plot(nums_episodes,discounted_rewards)
plt.savefig("Queue Learning")
plt.show()

print(time.time()-start)

#### Q Learning with decaying exploration rate ####
print("Q Learning with decaying exploration rate")
start = time.time()

discounted_rewards = []
for num_episodes in nums_episodes:
    averaged_sum = 0
    for i in range(batch_size):
        policy = {state : 0 for state in mdp.states}
        learned_policy, discounted_reward = policy_obj.q_learning(mdp, policy, alpha, discount,epsilon, num_episodes = num_episodes,decaying_epsilon=True)
        averaged_sum += discounted_reward
    averaged_sum /= batch_size
    discounted_rewards.append(discounted_reward)
plt.title("Discounted reward sum vs no. of episodes (Queue Learning with decaying exploration rate)" )
plt.plot(nums_episodes,discounted_rewards)
plt.savefig("Queue Learning 2")
plt.show()

print(time.time()-start)

#### SARSA ####
print("SARSA")
start = time.time()

discounted_rewards = []
for num_episodes in nums_episodes:
    averaged_sum = 0
    for i in range(batch_size):
        policy = {state : 0 for state in mdp.states}
        learned_policy, discounted_reward = policy_obj.SARSA(mdp, policy, alpha, discount,epsilon, num_episodes = num_episodes)
        averaged_sum += discounted_reward
    averaged_sum /= batch_size
    discounted_rewards.append(discounted_reward)
plt.title("Discounted reward sum vs no. of episodes (SARSA)" )
plt.plot(nums_episodes,discounted_rewards)
plt.savefig("SARSA")
plt.show()

print(time.time()-start)

#### SARSA with decaying exploration rate ####
print("SARSA with decaying exploration rate")
start = time.time()

discounted_rewards = []
for num_episodes in nums_episodes:
    averaged_sum = 0
    for i in range(batch_size):
        policy = {state : 0 for state in mdp.states}
        learned_policy, discounted_reward = policy_obj.SARSA(mdp, policy, alpha, discount,epsilon, num_episodes = num_episodes,decaying_epsilon=True)
        averaged_sum += discounted_reward
    averaged_sum /= batch_size
    discounted_rewards.append(discounted_reward)
plt.title("Discounted reward sum vs no. of episodes (SARSA with decaying exploration rate)" )
plt.plot(nums_episodes,discounted_rewards)
plt.savefig("SARSA 2")
plt.show()

print(time.time()-start)


