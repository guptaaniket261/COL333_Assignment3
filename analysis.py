import A3
import time
import matplotlib.pyplot as plt


instance = A3.Taxi_MDP()
policy_obj = A3.Policy()

##### VALUE ITERATION #####

## Choosing epsilon ##
# eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-18]
# for epsilon in eps:
#     policy = policy_obj.value_iteration(instance, epsilon, 0.9)

## Discount Factor and Rate of convergence ##
# discounts = [0.01, 0.1, 0.5, 0.8, 0.99]
# for discount in discounts:
#     policy = policy_obj.value_iteration(instance, 1e-18, discount)

## Part A1-3 ##
# T,P,D = (0,0),(4,0),(0,4)
# instance = A3.Taxi_MDP(T = T, P_pos = P, D = D)
# epsilon = 1e-18
# policy_1 = policy_obj.value_iteration(instance,epsilon,0.1)
# #instance.simulate(policy_1)
# policy_2 = policy_obj.value_iteration(instance,epsilon,0.99)
# instance.simulate(policy_2)


alpha = 0.6
discount = 1.0
epsilon = 0.1
#batch_size = 20
#nums_episodes = [2000, 2500, 3000, 3500, 4000, 4500]
#nums_episodes = [2000,2200,2400,2600,2800,3000]
policy = {state : 0 for state in instance.states}
print(instance.startState)
print(instance.currState)
print(instance.destState)
learned_policy, discounted_reward = policy_obj.q_learning(instance,policy,alpha,discount,epsilon)
instance.simulate(learned_policy)


'''
#### Q Learning ####
print("Q Learning")
start = time.time()

discounted_rewards = []
for num_episodes in nums_episodes:
    averaged_sum = 0
    for i in range(batch_size):
        # instance.get_rand_start()
        policy = {state : 0 for state in instance.states}
        learned_policy, discounted_reward = policy_obj.q_learning(instance, policy, alpha, discount,epsilon, num_episodes = num_episodes)
        instance.simulate(learned_policy)
        averaged_sum += discounted_reward
    averaged_sum /= batch_size
    discounted_rewards.append(averaged_sum)
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
        policy = {state : 0 for state in instance.states}
        learned_policy, discounted_reward = policy_obj.q_learning(instance, policy, alpha, discount,epsilon, num_episodes = num_episodes,decaying_epsilon=True)
        averaged_sum += discounted_reward
    averaged_sum /= batch_size
    discounted_rewards.append(averaged_sum)
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
        policy = {state : 0 for state in instance.states}
        learned_policy, discounted_reward = policy_obj.SARSA(instance, policy, alpha, discount,epsilon, num_episodes = num_episodes)
        averaged_sum += discounted_reward
    averaged_sum /= batch_size
    discounted_rewards.append(averaged_sum)
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
        policy = {state : 0 for state in instance.states}
        learned_policy, discounted_reward = policy_obj.SARSA(instance, policy, alpha, discount,epsilon, num_episodes = num_episodes,decaying_epsilon=True)
        averaged_sum += discounted_reward
    averaged_sum /= batch_size
    discounted_rewards.append(averaged_sum)
plt.title("Discounted reward sum vs no. of episodes (SARSA with decaying exploration rate)" )
plt.plot(nums_episodes,discounted_rewards)
plt.savefig("SARSA 2")
plt.show()

print(time.time()-start)

'''
