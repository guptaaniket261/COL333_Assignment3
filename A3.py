import numpy as np
import matplotlib.pyplot as plt
import time

def goLeft(r,c,bigger_grid):
	bannedLeft = [(0,2),(1,2),(3,1),(3,3),(4,1),(4,3)]
	if bigger_grid:
		bannedLeft = [ (0,3),(1,3),(2,3),(3,3),(0,8),(1,8),(2,8),(3,8),(2,6),(3,6),(4,6),(5,6),(6,1),(7,1),(8,1),(9,1),(6,4),(7,4),(8,4),(9,4),(6,8),(7,8),(8,8),(9,8)  ]
	if (r,c) in bannedLeft:
		return False
	return True

def goRight(r,c,bigger_grid):
	bannedRight = [(0,1),(1,1),(3,0),(3,2),(4,0),(4,2)]
	if bigger_grid:
		bannedRight = [(0,2),(1,2),(2,2),(3,2),(0,7),(1,7),(2,7),(3,7),(2,5),(3,5),(4,5),(5,5),(6,0),(7,0),(8,0),(9,0),(6,3),(7,3),(8,3),(9,3),(6,7),(7,7),(8,7),(9,7) ]
	if (r,c) in bannedRight:
		return False
	return True

def encode(row_p, col_p, row_t, col_t, inside_taxi):
	return (10000*row_p + 1000*col_p + 100*row_t + 10*col_t+ inside_taxi)

def decode(state):
	# row_p, col_p, row_t, col_t, inside_taxi
	sol = [0,0,0,0,0]
	for i in range(5):
		sol[4-i] = (state % 10)
		state = state//10
	return sol

def equal(dict1, dict2):
	for i in dict1:
		if dict1[i] != dict2[i]:
			return False
	return True

# print(decode(encode(1,2,3,4,9)))

def evaluate(q_table, states, batch_size, discount):
	disc_sum_rewards = []
	learned_policy = {state : -1 for state in states}
	for state in states:
		learned_policy[state] = max(q_table[state], key= lambda action: q_table[state][action])
	# print(learned_policy)
	for i in range(batch_size):
		test_mdp = Taxi_MDP() 
		#test_mdp.get_rand_start()
		rewards = test_mdp.simulate(learned_policy,verbose = False)
		disc_sum_reward = 0
		factor = 1
		for j in range(len(rewards)):
			disc_sum_reward += factor*rewards[j]
			factor *= discount
		disc_sum_rewards.append(disc_sum_reward)
	print(disc_sum_reward)
	averaged_sum = sum(disc_sum_rewards) / batch_size
	return averaged_sum

class State:

	def __init__(self, row_p, col_p, row_t, col_t, inside_taxi):
		self.row_p = row_p
		self.col_p = col_p
		self.row_t = row_t
		self.col_t = col_t
		self.inside_taxi = inside_taxi


class Taxi_MDP:
	"""
	5x5 grid 
  +---------+
  |R: | : :G|
  |T: | : : |
  | : : : : |
  | | : | : |
  |Y| : |B: |
  +---------+

  """
	# (0, 0) is at top left
  # Taxi Start Position, T = (x,y)
  # Passenger Initial position, P_pos = (a,b)
  # Passenger Destination, D = (p,q)

	# actions = 0: South
	#           1: North
	#           2: East
	#           3: West
	#           4: Pickup
	#           5: Putdown

	def __init__(self, T = (1,0), P_pos = (0,0), D = (0,4), bigger_grid = False):
		self.p_locs = [(0,0),(0,4),(4,0),(4,3)]
		self.num_rows = 5
		self.num_cols = 5
		self.bigger_grid = bigger_grid

		if bigger_grid:
			self.p_locs = [ (0,0),(0,5),(0,8),(3,3),(4,6),(8,0),(9,4),(9,9) ]
			self.num_rows = 10
			self.num_cols = 10
			T, P_pos, D = (5,2), (0,0), (0,8)

		if P_pos not in self.p_locs or D not in self.p_locs:
			print("Starting and ending position must be among the given depots")
			raise 0

		self.currState = encode(P_pos[0], P_pos[1], T[0], T[1], 0)
		self.startState = encode(P_pos[0], P_pos[1], T[0], T[1], 0)	
		self.destState = encode(D[0], D[1], D[0], D[1], 0)
		self.A = {"S": 0, "N": 1, "E": 2, "W": 3, "Pickup": 4, "Putdown": 5}
		self.states, self.state_index, self.index_state = self.getStates()
		self.P, self.R = self.getModels()

	def getStates(self):
		states = []
		state_index = {}
		index_state = {}
		counter = 0
		for row_p in range(self.num_rows):
			for col_p in range(self.num_cols):
				state = encode(row_p, col_p, row_p, col_p, 1)
				states.append(state)
				state_index[state] = counter
				index_state[counter] = state
				counter += 1
				for row_t in range(self.num_rows):
					for col_t in range(self.num_cols):
						state = encode(row_p, col_p, row_t, col_t, 0)
						states.append(state)
						state_index[state] = counter
						index_state[counter] = state
						counter += 1
		return states, state_index, index_state

	def getModels(self):
		P = dict()
		R = dict()
		states = self.states
		
		num_actions = len(self.A)
		num_rows = self.num_rows
		num_cols = self.num_cols
		for state in states:
			P[state] = {action: [] for action in range(num_actions)}
			R[state] = {action: 0 for action in range(num_actions)}
		for state in states:
			if state == self.destState:
				continue
			row_p, col_p, row_t, col_t, inside_taxi = decode(state)
			n_row0, n_row1, n_row2, n_row3 = min(row_t+1,num_rows-1), max(row_t-1, 0), row_t, row_t
			n_col0, n_col1, n_col2, n_col3 = col_t, col_t, col_t, col_t
			if goRight(row_t, col_t,self.bigger_grid):
				n_col2 = min(col_t+1,num_cols-1)
			if goLeft(row_t,col_t,self.bigger_grid):
				n_col3 = max(col_t-1,0)

			if (inside_taxi == 1):
				new_state0 = encode(n_row0, n_col0, n_row0, n_col0, inside_taxi)
				new_state1 = encode(n_row1, n_col1, n_row1, n_col1, inside_taxi)
				new_state2 = encode(n_row2, n_col2, n_row2, n_col2, inside_taxi)
				new_state3 = encode(n_row3, n_col3, n_row3, n_col3, inside_taxi)

			else:
				new_state0 = encode(row_p, col_p, n_row0, n_col0, inside_taxi)
				new_state1 = encode(row_p, col_p, n_row1, n_col1, inside_taxi)
				new_state2 = encode(row_p, col_p, n_row2, n_col2, inside_taxi)
				new_state3 = encode(row_p, col_p, n_row3, n_col3, inside_taxi)
		
			for action in range(num_actions):
				if action < 4:
					same_state = False
					same_state_prob = 0
					R[state][action] = -1
					if action == 0:
						if new_state0 == state:
							same_state = True
							same_state_prob += 0.85
						else:
							P[state][action].append((0.85,new_state0))
						if new_state1 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state1))
						if new_state2 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state2))
						if new_state3 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state3))
						if same_state:
							P[state][action].append((same_state_prob,state))
						
					if action == 1:
						if new_state1 == state:
							same_state = True
							same_state_prob += 0.85
						else:
							P[state][action].append((0.85,new_state1))
						if new_state2 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state2))
						if new_state3 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state3))
						if new_state0 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state0))
						if same_state:
							P[state][action].append((same_state_prob,state))

					if action == 2:
						if new_state2 == state:
							same_state = True
							same_state_prob += 0.85
						else:
							P[state][action].append((0.85,new_state2))
						if new_state3 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state3))
						if new_state0 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state0))
						if new_state1 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state1))
						if same_state:
							P[state][action].append((same_state_prob,state))

					if action == 3:
						if new_state3 == state:
							same_state = True
							same_state_prob += 0.85
						else:
							P[state][action].append((0.85,new_state3))
						if new_state0 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state0))
						if new_state1 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state1))
						if new_state2 == state:
							same_state = True
							same_state_prob += 0.05
						else:
							P[state][action].append((0.05,new_state2))
						if same_state:
							P[state][action].append((same_state_prob,state))

				else:
					if not(row_t == row_p and col_t == col_p):
						R[state][action] = -10
						P[state][action].append((1.0, state))
						continue

					if action == 4:
						n_state = encode(row_p, col_p, row_t, col_t, 1)
					else:
						n_state = encode(row_p, col_p, row_t, col_t, 0)
					P[state][action].append((1.0, n_state))
					if n_state == self.destState:
						# TODO
						R[state][action] = 20
					else:
						R[state][action] = -1
		return P, R

	def step(self, action):
		if self.currState == self.destState:
			print("Destination Reached...")
			return self.currState, 0

		transitions = self.P[self.currState][action]
		n_states = len(transitions)
		probs = [100*i[0] for i in transitions]
		cum_probs = np.cumsum(probs)
		r = np.random.randint(low = 0, high = 100)
		for i in range(n_states):
			if r < cum_probs[i]:
				new_state = transitions[i][1] 
				# print(self.currState, transitions)
				reward = self.R[self.currState][action]
				self.currState = transitions[i][1]
				
				# print(decode(new_state))
				#return transitions[i][0], reward
				return transitions[i], reward ########## CHANGED: CHANGE1 #############

	def simulate(self, policy,verbose=True):
		steps = 0
		rewards = []
		A = {0 :"S", 1: "N", 2: "E", 3: "W", 4: "Pickup", 5: "Putdown"}
		while(self.currState !=self.destState):
			if steps > 500:
				break
			steps += 1
			prev_state = self.currState
			prev_Px, prev_Py, prev_Tx, prev_Ty, inside_taxi = decode(prev_state)
			prev_P, prev_T = (prev_Px, prev_Py), (prev_Tx, prev_Ty)
			transition, reward = self.step(policy[prev_state])  ### transition = prob, next_state
			prob = transition[0]   ## CHANGE1 
			new_state = self.currState
			new_Px, new_Py, new_Tx, new_Ty, n_inside_taxi = decode(new_state)
			new_P , new_T = (new_Px, new_Py), (new_Tx, new_Ty)
			action_taken = A[policy[prev_state]]
			if verbose:
				print("CurrTaxi: {0}, CurrPass: {1}, Inside:{2}, action: {3}, prob: {4}, reward: {5}, NewTaxi: {6}, NewPass: {7}, Inside:{8}".format(prev_T, prev_P, inside_taxi, action_taken, prob, reward,new_T, new_P, n_inside_taxi ))
				#print("CurrT: {0}, CurrP: {1}, In:{2}, A: {3}, P: {4}, R: {5}, NewT: {6}, NewP: {7}, In:{8}".format(prev_T, prev_P, inside_taxi, action_taken, prob, reward,new_T, new_P, n_inside_taxi ))
			rewards.append(reward)
		return rewards

	def get_rand_start(self):
		row_pd,col_pd, row_td,col_td,inside_taxi_d = decode(self.destState)
		row_p,col_p = row_pd,col_pd
		while row_p == row_pd and col_p == col_pd:
			r = np.random.randint(low = 0, high = 4)
			row_p, col_p = self.p_locs[r]
		row_t = np.random.randint(low = 0, high = 5)
		col_t = np.random.randint(low = 0, high = 5)	
		inside_taxi = 0
		start_state = encode(row_p, col_p, row_t, col_t, inside_taxi)
		self.startState = start_state
		self.currState = start_state
		return start_state

class Policy:
	def __init__(self):
		return

	def value_iteration(self, MDP, epsilon, discount=0.9):
		old_utilities = {state : 0 for state in MDP.states}
		new_utilities = {state : 0 for state in MDP.states}
		delta = 0
		converged = False
		policy = {state : -1 for state in MDP.states}
		row_p, col_p, row_t, col_t, inside_taxi = decode(MDP.destState)
		pre_destination = encode(row_p, col_p, row_t, col_t, 1)
		# print(pre_destination)
		iterations = 0
		iter_array = []
		max_norm_array = []
		while(not converged):
			iterations += 1
			# print("Dest state utility: ", old_utilities[MDP.destState])
			# print("PreDest state utility: ", old_utilities[pre_destination])
			for state in MDP.states:
				old_utilities[state] = new_utilities[state]
			delta = 0
			for state in MDP.states:
				new_utilities[state] = -np.inf
				for a in MDP.A:
					action = MDP.A[a]
					temp_utility = 0
					for prob, neighbour in MDP.P[state][action]:
						temp_utility += prob*(MDP.R[state][action] + discount*old_utilities[neighbour])
					new_utilities[state] = max(temp_utility, new_utilities[state])
				delta = max(delta, abs(new_utilities[state] - old_utilities[state]))
			max_norm_array.append(delta)
			iter_array.append(iterations)
			if delta < epsilon:
				converged = True
		plt.plot(iter_array, max_norm_array)
		plt.title("Value Iteration, epsilon = " + str(epsilon) + ", discount = "+str(discount) )
		plt.xlabel("Number of iterations")
		plt.ylabel("Max Norm value")
		plt.savefig("Val_iter_d_" + str(int(discount*100)))
		plt.show()
		
		for state in MDP.states:
			optimal_utility = -np.inf
			for a in MDP.A:   ## MDP.A = {"S": 0, "N": 1, "E": 2, "W": 3, "Pickup": 4, "Putdown": 5}
				action = MDP.A[a]
				temp_utility = 0
				for prob, neighbour in MDP.P[state][action]:
					temp_utility += prob*(MDP.R[state][action] + discount*old_utilities[neighbour])
				if temp_utility > optimal_utility:
					optimal_utility = temp_utility
					policy[state] = action

		
		print("Epsilon: {0}, Discount: {1} => Iterations: {2}".format(epsilon, discount,iterations))
		return policy

	def policy_evaluation_linear(self, MDP, policy, discount):
		num_states = len(MDP.states)
		equations_LHS = [ [] for i in range(num_states) ]
		equations_RHS = [0 for i in range(num_states)]
		for state in MDP.states:
			action = policy[state]
			equation = [ 0 for j in range(num_states)]
			equation[MDP.state_index[state]] = 1
			for prob, next_state in MDP.P[state][action]:
				equation[MDP.state_index[next_state]] = - prob * discount
			equations_LHS[MDP.state_index[state]] = equation
			equations_RHS[MDP.state_index[state]] = MDP.R[state][action]
		vals = np.linalg.solve(equations_LHS,equations_RHS)
		utilities = {MDP.index_state[i]: v for i,v in enumerate(vals)}
		return utilities

	def policy_evaluation_iterative(self, MDP, policy, discount, epsilon = 0.001):
		converged = False
		delta = 0
		old_utilities = {state : 0 for state in MDP.states}
		new_utilities = {state : 0 for state in MDP.states}
		while(not converged):
			delta = 0
			for state in MDP.states:
				old_utilities[state] = new_utilities[state]
			for state in MDP.states:
				action = policy[state]
				temp_utility = 0
				for prob, neighbour in MDP.P[state][action]:
					temp_utility += prob*(MDP.R[state][action] + discount*old_utilities[neighbour])
				new_utilities[state] = temp_utility
				delta = max(delta, abs(new_utilities[state] - old_utilities[state]))
			if delta < epsilon:
				converged = True
		return new_utilities

	def policy_iteration(self, MDP, temp_policy, discount, epsilon = 0.001, iterative = 0):
		utilities = {state : 0 for state in MDP.states}
		improved_policy = temp_policy
		converged = False
		iteration = 0
		while(not converged):
			policy = {state : improved_policy[state] for state in MDP.states}
			iteration += 1
			print("Iteration: " + str(iteration))
			## Policy Evaluation ##
			if iterative == 1:
				utilities = self.policy_evaluation_iterative(MDP, policy, discount, epsilon)
			else:
				utilities = self.policy_evaluation_linear(MDP, policy, discount)

			## Policy Improvement ##
			for state in MDP.states:
				optimal_utility = -np.inf
				for a in MDP.A:
					action = MDP.A[a]
					temp_utility = 0
					for prob, neighbour in MDP.P[state][action]:
						temp_utility += prob*(MDP.R[state][action] + discount*utilities[neighbour])
					if temp_utility > optimal_utility:
						optimal_utility = temp_utility
						improved_policy[state] = action
			
			# if iteration< 3:
			# 	print(policy)
			# 	print(improved_policy)
			if equal(policy , improved_policy) or iteration > 50:
				# print(policy)
				# print(improved_policy)
				converged = True			
		
		return utilities,improved_policy

	def epsilon_greedy(self,action,epsilon, iter_num, decaying_epsilon):
		r = np.random.rand()
		if decaying_epsilon:
			epsilon = epsilon/iter_num
		if r < epsilon:
			## Exploration
			action = np.random.randint(low=0,high=6)
		return action

	
	def getRandStart(self, destination):
		row_t = np.random.randint(low = 0, high = 5)
		col_t = np.random.randint(low = 0, high = 5)
		p_locs = [(0,0),(0,4),(4,0),(4,3)]
		found = False
		loc = (0, 0)
		while( not found):
			r = np.random.randint(low = 0, high = 4)
			loc = p_locs[r]
			if loc!=destination:
				break
		return loc, (row_t, col_t)


	def q_learning(self,MDP,policy,alpha,discount,epsilon,num_episodes=500,decaying_epsilon = False,max_steps = 500):
		# print(policy)
		q_table = {state: {action: 0 for action in range(6)} for state in MDP.states}
		iteration = 1
		rewards, episodes = [], []
		for episode in range(num_episodes):
			# state = MDP.get_rand_start() 
			dest = (0, 4)
			p_loc, t_loc = self.getRandStart(dest)
			MDP = Taxi_MDP(t_loc, p_loc, dest)
			# MDP.currState = MDP.startState
			state = MDP.startState
			done = False
			num_steps = 0
			while (not done):
				if num_steps>max_steps:
					print(episode)
					break
				action = self.epsilon_greedy(policy[state], epsilon, iteration, decaying_epsilon)
				#print(action)
				transition, reward = MDP.step(action)   ## transition = prob, next_state
				next_state = transition[1]   ## same as MDP.currState
				#print(next_state,reward)
				# next_opt_action = max(q_table[next_state], key= lambda action: q_table[next_state][action])
				next_opt_action = policy[next_state]
				# print(q_table)
				#print(next_opt_action)
				td_update_sample = reward + discount * q_table[next_state][next_opt_action] 
				q_table[state][action] = (1-alpha) * q_table[state][action] +  alpha * td_update_sample
				iteration += 1
				if next_state == MDP.destState:
					done = True
					print("Reached")
				state = next_state
				num_steps += 1
				# for state in MDP.states:
					# policy[state] = max(q_table[state], key= lambda action: q_table[state][action])
				policy[state] = max(q_table[state], key= lambda action: q_table[state][action])
				
			
			####   Calculate discounted sum of rewards for this episode, averaged over 10 runs  #####
			curr_score = evaluate(q_table,MDP.states,10,discount)
			rewards.append(curr_score)  
			episodes.append(episode)

		plt.plot(episodes, rewards)
		plt.show()
		learned_policy = {state : -1 for state in MDP.states}
		for state in MDP.states:
			learned_policy[state] = max(q_table[state], key= lambda action: q_table[state][action])
		utility = evaluate(q_table,MDP.states,10,discount)
		print(utility)		
		return learned_policy, utility


	def SARSA(self,MDP,policy,alpha,discount,epsilon,num_episodes=2000,decaying_epsilon = False,max_steps=500):
		q_table = {state: {action: 0 for action in range(6)} for state in MDP.states}
		iteration = 1
		rewards = []
		episodes = []
		for episode in range(num_episodes):
			state = MDP.get_rand_start() 
			done = False
			num_steps = 0
			while (not done) and num_steps < max_steps:
				action = self.epsilon_greedy(policy[state], epsilon, iteration, decaying_epsilon)
				transition, reward = MDP.step(action)   ## transition = prob, next_state
				next_state = transition[1]
				next_action = policy[next_state]
				td_update_sample = reward + discount * q_table[next_state][next_action] 
				q_table[state][action] = (1-alpha) * q_table[state][action] +  alpha * td_update_sample
				iteration += 1
				if next_state == MDP.destState:
					done = True
				state = next_state
				num_steps+=1
				for state in MDP.states:
					policy[state] = max(q_table[state], key= lambda action: q_table[state][action])

			##### Calculate discounted sum of rewards for this episode, averaged over 10 runs #####
			# curr_score = evaluate(q_table,MDP.states,10,discount)
			# rewards.append(curr_score)  
			# episodes.append(episode)			

		learned_policy = {state : -1 for state in MDP.states}
		for state in MDP.states:
			learned_policy[state] = max(q_table[state], key= lambda action: q_table[state][action])
		return learned_policy
		

def main():
	# a = Taxi_MDP()
	# po = Policy()
	# eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-18]
	# for epsilon in eps:
	# 	policy = po.value_iteration(a, epsilon, 0.9)

	# discounts = [0.01, 0.1, 0.5, 0.8, 0.99]
	# for discount in discounts:
	# 	policy = po.value_iteration(a, 1e-18, discount)
	
	T , P, D = (3,0), (0,0), (0,4)
	instance = Taxi_MDP(T, P, D)
	policy_finder = Policy()
	# print(instance.startState)


	# policy = policy_finder.value_iteration(instance, 1e-18, 0.1)
	# # print(policy[instance.startState])
	# # print(instance.P[instance.startState][policy[instance.startState]])
	# instance.simulate(policy)



	# utilities, policy = policy_finder.policy_iteration(instance, temp_policy, 0.99, 1e-8, 1)
	# instance.simulate(policy)

	## Evaluating the episodes while learning ## (remove) 

	#### Q Learning ####
	print("Q Learning")
	start = time.time()
	temp_policy = {state : np.random.randint(low = 0 , high = 6) for state in instance.states}	
	policy = policy_finder.q_learning(instance,temp_policy,0.25,0.99,0.1)
	print(policy)
	instance.simulate(policy[0])

	print(time.time()-start)

	# #### Q Learning with decaying exploration rate ####
	# print("Q Learning with decaying exploration rate")
	# start = time.time()
	# temp_policy = {state : 0 for state in instance.states}	
	# policy = policy_finder.q_learning(instance,temp_policy,0.25,0.99,1e-18,decaying_epsilon=True)
	# print(time.time()-start)
	
	# #### SARSA ####
	# print("SARSA")
	# start = time.time()
	# temp_policy = {state : 0 for state in instance.states}	
	# policy = policy_finder.SARSA(instance,temp_policy,0.25,0.99,1e-18)
	# print(time.time()-start)

	# #### SARSA with decaying exploration rate ####
	# print("SARSA with decaying exploration rate")
	# start = time.time()
	# temp_policy = {state : 0 for state in instance.states}	
	# policy = policy_finder.SARSA(instance,temp_policy,0.25,0.99,1e-18,decaying_epsilon=True)
	# print(time.time()-start)


	# print(policy)
	# a.step(0)
	# a.step(1)
	
if __name__ == "__main__":
    main()
					
					










		


		

