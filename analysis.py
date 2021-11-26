import A3
import time
import matplotlib.pyplot as plt




def main():
    instance = A3.Taxi_MDP()
    policy_obj = A3.Policy()
    ques = input("Enter Part (A or B), Question number and part number. Example: A 3 b\n").strip().split()
    part = ques[0]
    ques_num = ques[1]
    sub_part = ques[2]
    if part == "A":
        if ques_num == "2":
            if sub_part == "a":
                policy = policy_obj.value_iteration(instance,1e-6,0.9)
                instance.simulate(policy)
            if sub_part == "b":
                # Discount Factor and Rate of convergence ##
                discounts = [0.01, 0.1, 0.5, 0.8, 0.99]
                for discount in discounts:
                    policy = policy_obj.value_iteration(instance, 1e-18, discount)
            if sub_part == "c":
                T,P,D = (0,0),(4,0),(0,4)
                instance = A3.Taxi_MDP(T = T, P_pos = P, D = D)
                epsilon = 1e-18
                policy_1 = policy_obj.value_iteration(instance,epsilon,0.1)
                instance.simulate(policy_1)
                policy_2 = policy_obj.value_iteration(instance,epsilon,0.99)
                instance.simulate(policy_2)

                print("Random Start State")
                instance.get_rand_start()
                policy_1 = policy_obj.value_iteration(instance,epsilon,0.1)
                instance.simulate(policy_1)
                instance.get_rand_start()
                policy_2 = policy_obj.value_iteration(instance,epsilon,0.99)
                instance.simulate(policy_2)

        if ques_num == "3":
            if sub_part == "a":
                discount = 0.9
                print("Policy Iteration using Iterative Evaluation Method ")
                policy = {state: 0 for state in instance.states}
                utilities, learned_policy = policy_obj.policy_iteration(instance,policy, discount, iterative=1)
                instance.simulate(learned_policy)

                print("Policy Iteration using Linear Algebraic Evaluation Method ")
                policy = {state: 0 for state in instance.states}
                utilities, learned_policy = policy_obj.policy_iteration(instance,policy, discount)
                instance.simulate(learned_policy)

            if sub_part == "b":
                discounts = [0.01, 0.1, 0.5, 0.8, 0.99]
                for discount in discounts:
                    policy = {state: 0 for state in instance.states}
                    utilities, learned_policy = policy_obj.policy_iteration(instance,policy, discount, iterative=1)
                    instance.simulate(learned_policy)
                    policy = {state: 0 for state in instance.states}
                    utilities_after_eval,l_policy = policy_obj.policy_iteration(instance,policy,discount,iterative=1,calc_loss = True,opt_utilities = utilities)

main()





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

##### POLICY ITERATION #####
'''
discounts = [0.01, 0.1, 0.5, 0.8, 0.99]
for discount in discounts:
    policy = {state: 0 for state in instance.states}
    utilities, learned_policy = policy_obj.policy_iteration(instance,policy, discount, iterative=1)
    instance.simulate(learned_policy)
    policy = {state: 0 for state in instance.states}
    utilities_after_eval,l_policy = policy_obj.policy_iteration(instance,policy,discount,iterative=1,calc_loss = True,opt_utilities = utilities)

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
