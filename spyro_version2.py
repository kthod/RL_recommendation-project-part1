 
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt 
# Environment parameters
K = 15  # Number of content items
u_min = 0.1  # Threshold for content relevance
C = int(0.2 * K)  # Number of cached items
#C = 2
# User model parameters
N = 2 # Number of recommended items
q = 0.2  # Probability of ending the viewing session
alpha = 0.9  # Probability of selecting a recommended item
tradeoff_factor=0.4
# Generate random relevance values
U = np.random.rand(K, K)
np.fill_diagonal(U, 0)  # Set diagonal elements to 0
# U = np.array([[0., 0.09709217, 0.95697935, 0.76421269, 0.79379138],
#               [0.85679266, 0., 0.73115609, 0.97025111, 0.00706508],
#               [0.38327773, 0.27582305, 0., 0.40938946, 0.70918518],
#               [0.27415892, 0.89691232, 0.47103534, 0., 0.97776446],
#               [0.06699551, 0.96500574, 0.00547615, 0.74654658, 0.]])
# U = np.array([[0.0, 0.8, 0.3, 0.6, 0.1],
#               [0.8, 0.0, 0.7, 0.4, 0.3],
#               [0.3, 0.7, 0.0, 0.2, 0.9],
#               [0.6, 0.4, 0.2, 0.0, 0.8],
#               [0.1, 0.3, 0.9, 0.8, 0.0]])
print(U)
#vector to denote the cost of each state. 1 for non-cached, 0 for cached
Cost = [1]*(K-C) +[0]*C   
# Cost = [1,0,1,0,1]
random.shuffle(Cost)
print(Cost)

Tmax = 10000

#create action set as the set of every possible combination of N=2 states 
action_set = []
for i in range(K):
    for j in range(i+1,K):
        a = (i, j)
        action_set.append(a)

num_of_actions = len(action_set)

def normalize_matrix(A):
    """
    function to normalize matrix such that the sum of each row is N
    """
    sum_a = np.sum(A,axis=1)
    for i in range(K):
        A[i] = N/sum_a[i]*A[i]

    return A

U_norm =normalize_matrix(U) #normalize matrix U such that the sum of each row is N



# def define_rewards():
#     Rewards = np.zeros((K,K))
#     for i in range(K):
#         for j in range(K):
#             Rewards[i][j] = tradeoff_factor*U_norm[i][j] + (1-tradeoff_factor)*( 1-Cost[j] )
#     return Rewards

# Rewards =define_rewards()

def all_recommendations_are_relevant(recommendations,s):
    """
    function to check whether everey recommended state in the racommendation batch is 
    relevant to s

    arguments:
    recommendations (tuple of ints): recommendation batch for state s
    s (int): current state

    returns:
    True: if all recommendation are relevant to s
    False: otherwise
    """
    for u in recommendations:
        if U[s][int(u)]<u_min:
            return False
    return True

def get_next_states(s, a): 
    """
    function to generate all possible next states alongiside their respective probabilities and rewards
    given a state s and an action a

    arguments:
    s (int):  current state
    a (tuple of ints): recommendation batch for state s

    returns:
    next_states (list of tuples):a list of tuples, where each tuple denotes the next state alongiside its respective probability and reward
    """
    next_states = []
    if(all_recommendations_are_relevant(a, s)): #determine the probabilities for next given that all recommendation are relevant
        for s_prime in range(K):
                if s_prime in a:
                    prob = (alpha/N + (1-alpha)/K) #probability to choose a video from the recommedation batch 
                    reward = 1-Cost[s_prime] #define reward to be 1 for cached and 0 for uncached video
                    next_states.append((prob,s_prime,reward))
                else:  
                    prob = (1-alpha)/K #probability to choose a video outside the recommedation batch 
                    reward = 1-Cost[s_prime]
                    next_states.append((prob,s_prime,reward))
            
    else: #determine the probabilities for next given that at least one recommendation is irrelevant
        for s_prime in range(K):
                
                prob = 1/K 
                reward = 1-Cost[s_prime]
                next_states.append((prob,s_prime,reward))
                
    return next_states
       
def policy_evaluation(pi, gamma = 1.0, epsilon = 1e-10):  
    """
    Function to evaluate a given policy

    arguments:
    pi (list of tuples): policy to be evaluated
    gamma (float): discounting factor
    epsilon (float): approximation tolerance

    returns:
    V (vector of floats): vector of value function of each state calculated using bellman equation
    """

    prev_V = np.zeros(K) # use as "cost-to-go", i.e. for V(s')
    while True: #performing iterations
        V = np.zeros(K) # current value function to be learnerd
        for s in range(K):  # do for every state
            for prob, next_state, reward in get_next_states(s, pi[s]):  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                V[s] += prob * (reward + gamma * prev_V[next_state] )
        if np.max(np.abs(prev_V - V)) < epsilon: #check if the new V estimate is close enough to the previous one; 
            break # if yes, finish loop
        prev_V = V.copy() #freeze the new values (to be used as the next V(s'))
    return V





def policy_improvement(V, gamma=1.0):  
    """
    Function to greedily improve policy

    arguments:
    V (vector of floats): V (vector of floats): vector of value function of each state calculated using bellman equation
    gamma (float): discounting factor
    
    returns:
    new_pi (list of tuples): new policy
    """
    Q = np.zeros((K, len(action_set)), dtype=np.float64) #create a Q value array
    for s in range(K):        # for every state in the environment/model
        for i,a in enumerate(action_set):  # and for every action in that state
            for prob, next_state, reward in get_next_states(s, a):  #evaluate the action value based on the model and Value function given (which corresponds to the previous policy that we are trying to improve) 
                Q[s][i] += prob * (reward + gamma * V[next_state] )
#     new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]  # this basically creates the new (improved) policy by choosing at each state s the action a that has the highest Q value (based on the Q array we just calculated)
    new_pi = np.zeros((K, N), dtype=np.int8)
    for s in range(K):
        
        while True:
            best_action = np.argmax(Q[s])
            if s not in action_set[best_action]:
                break
            Q[s][best_action]=float('-inf')

        new_pi[s] = action_set[best_action]
    
    return new_pi


def policy_iteration( gamma = 1.0, epsilon = 1e-10):
    """
    Function to converge to the optimal policy by performing iterative avaluation and improvments until convergence

    arguments:
    gamma (float): discounting factor
    epsilon (float): approximation tolerance

    returns:
    pi (list of tuples): optimal policy
    V (vector of floats): vector of value function of each state calculated using bellman equation
    """
    t = 0
    # random_actions = np.random.choice(tuple(P[0].keys()), len(P))     # start with random actions for each state  
    # pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]     # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)
    
    pi = np.random.randint(0, K, size=(K, N))  # Random initial policy
    #print(np.sum(R,axis=1))

    while True:
        old_pi = pi.copy()  #keep the old policy to compare with new
        V = policy_evaluation(pi,gamma,epsilon)   #evaluate latest policy --> you receive its converged value function

        pi = policy_improvement(V,gamma)          #get a better policy using the value function of the previous one just calculated 
        
        t += 1

        if compare_two_policies(pi,old_pi) or t>Tmax: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
            break
    print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
   
    return V,pi
    
def compare_two_policies(pi,old_pi):
    """
    Function to compare if two policies are identical

    arguments:
    pi: current policy
    old: previous policy

    returns:
    True if they are identical
    Flase otherwise
    """
    for k in range(K):
        for n in range(N):
            if (pi[k][n]!=old_pi[k][n]):
                return False

    return True

def policy_evaluation_spyro(R, gamma = 1.0, epsilon = 1e-10):  
    """
    Function to evaluate a given policy

    arguments:
    R (K x K matrix): Rij denotes the probalility of state j being in the recommendation batch of i
    gamma (float): discounting factor
    epsilon (float): approximation tolerance

    returns:
    V (vector of floats): vector of value function of each state calculated using bellman equation
    """

    t = 0   #there's more elegant ways to do this
    prev_V = np.zeros(K) # use as "cost-to-go", i.e. for V(s')
    while True:
        # print(t)
        V = np.zeros(K) # current value function to be learnerd

        for s in range(K):  # do for every state
            p=0
            for s_prime in range(K):  # do for every state:  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                P_s_sprime = calculate_transition_probabilities(R,s,s_prime)
                reward = 1-Cost[s_prime]
                V[s] +=  P_s_sprime * (reward+gamma * prev_V[s_prime])
            
            #print(p)
        if np.max(np.abs(prev_V - V)) < epsilon or t>Tmax: #check if the new V estimate is close enough to the previous one; 
            break # if yes, finish loop
        prev_V = V.copy() #freeze the new values (to be used as the next V(s'))
        t += 1
        # Vplot[:,t] = prev_V  # accounting for GUI  
    return V

def policy_improvement_spyro(V):  
    """
    Function to update the matrix R using a trade off between relevance and cost

    arguments:
    V (vector of floats): V (vector of floats): vector of value function of each state calculated using bellman equation
    
    returns:
    R (K x K matrix): Rij denotes the probalility of state j being in the recommendation batch of i
    """
    V_ext = np.tile(V,(K,1)) #create a K x K matrix where row i V_ext[i]=V
    np.fill_diagonal(V_ext, 0) 

    V_ext_norm = normalize_matrix(V_ext)
    R = tradeoff_factor*U_norm+(1-tradeoff_factor)*V_ext_norm
    
    return R

# policy iteration is simple, it will call alternatively policy evaluation then policy improvement, till the policy converges.

def policy_iteration_spyro( gamma = 1.0, epsilon = 1e-10):
    """
    Function to converge to the optimal policy by performing iterative evaluation and improvments until convergence

    arguments:
    gamma (float): discounting factor
    epsilon (float): approximation tolerance

    returns:
    pi (list of tuples): optimal policy
    V (vector of floats): vector of value function of each state calculated using bellman equation
    """
    t = 0
    # random_actions = np.random.choice(tuple(P[0].keys()), len(P))     # start with random actions for each state  
    # pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]     # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)
    
    pi = np.zeros((K, N), dtype=np.int8)  #initial policy
    V = np.zeros(K)
    #V_ext = np.zeros((K,K))
    R = np.random.rand(K, K)
    R=normalize_matrix(R)

    while True:
        #old_pi = pi.copy()  #keep the old policy to compare with new
        old_R = R.copy()
        V = policy_evaluation_spyro(R,gamma,epsilon)   #evaluate latest policy --> you receive its converged value function

        R = policy_improvement_spyro(V)
    
        t += 1
        
        if np.max(np.abs(old_R - R)) < epsilon or t>Tmax: #check if the new V estimate is close enough to the previous one; 
            break # if 
        # if compare_two_policies(pi,old_pi) or t>Tmax: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
        #     break
    print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
   
   #obtain the optimal policy  by selecting greedily the N states with highest R value for each state
    for i in range(K):
        pi[i] = get_N_max_values(R[i])
   

    return V,pi

def calculate_transition_probabilities(R,s,s_prime):
    """
    Function to calculate probablity P(s->s') using R matrix
    """
    P0 = N*u_min

    P_s_sprime = (1-P0)*((alpha/N)*R[s][s_prime]+(1-alpha)/K)+P0/K    
    return P_s_sprime


def get_N_max_values(Q):
    """
    Function that a vector Q and returns the indices of its N max values
    """
    sorted_Q = sorted(range(len(Q)), key=lambda x: Q[x], reverse=True)

    return sorted_Q[:N]
V_opt,P_opt1 = policy_iteration(1-q,0.001)   #just example of calling the various new functions we created.

V_opt,P_opt2 = policy_iteration_spyro(1-q,0.001)

chat_pi = np.array([[4,3],
 [3,0],
 [4,3],
 [4,1],
 [1,3]])

print("#############################")
print(V_opt)
print("#############################")
print(P_opt1)
#print(P_opt2)
#plt.show()

def Q_learning(gamma,delta, epsilon,learning_rate):
    """
    Function to perform Q learning algorithm.

    arguments:
    gamma (float): discounting factor
    delta (float):  approximation tolerance
    epsilon (float): epslilon greedy probability
    learning_rate (float): learning rate

    returns:
    Q (matrix K x num_of_actions):  martix of state action value function calculated using bellman equation
    """
    Q = np.zeros((K,num_of_actions)) 
    prev_Q = np.zeros((K,num_of_actions))
    t = 0
    while True:
        s = np.random.randint(K) #random initial state
        while True:
            if np.random.uniform() < epsilon:  # Explore if e(t) times
                while True : 
                    a = np.random.randint(num_of_actions) #choose random action
                    if s not in action_set[a]: #keep choosing randomly until s is not in action a.k.a until you don't recommend yourself
                        break
            else:  # Exploit 1-e(t) times
                
                while True:
                    a = np.argmax(Q[s]) #choose greedily the action with highest Q value
                    if s not in action_set[a]:
                        break
                    Q[s][a]=-100

            if (all_recommendations_are_relevant(action_set[a],s)):
            
                if np.random.uniform() < alpha:  # If all recommended items are relevant
                    s_prime = int(np.random.choice(action_set[a]))  # Pick a random item from relevant recommended items
                else:  # If at least one recommended item is not relevant
                    s_prime = np.random.randint(K)  # Pick a random item
            else:
                s_prime = np.random.randint(K)  # Pick a random item
        
            if np.random.uniform() < q: #if user opt to terminate session
                target = (1/2 - Cost[s_prime])
                Q[s][a] = prev_Q[s][a] + learning_rate * ( target - prev_Q[s][a] )
                break
            else:
                target = (1/2 - Cost[s_prime]) - Cost[s_prime] + gamma*np.max(prev_Q[s_prime])
            Q[s][a] = prev_Q[s][a] + learning_rate * ( target - prev_Q[s][a] )
            
            s = s_prime
        t+=1
        #epsilon = (t+1)**(-1/3)*(num_of_actions*math.log(t+1))**(1/3)
        epsilon = 0.1
        #learning_rate = learning_rate*(1/t)**(1/2)
        
        if (np.max(np.abs(prev_Q - Q)) < delta and t>1000*K) or t > 10000*K: #check if the new V estimate is close enough to the previous one;
            
            break # if yes, finish loop
        prev_Q = Q.copy()
    print(t)
    return Q


pi_Q_learning  =  np.zeros((K, N), dtype=np.int8)

Q = Q_learning(1-q,0.00001,1,0.001)

for s in range(K):
    #Q[s][s] = float('-inf')
    action = np.argmax(Q[s])
    pi_Q_learning[s] = action_set[action]

print(pi_Q_learning)


def simulate_session(policy, max_steps=1000):
    """
    Simulate a viewing session.
    The session starts with a random item and ends when the user decides to quit or after max_steps.
    The function returns the total cost of the session.
    """
    s = np.random.randint(K)  # Start with a random item
    cost_total = Cost[s]  # Initialize total cost with the cost of the first item
    for _ in range(max_steps):
        if np.random.uniform() < q:  # The user decides to quit
            break

        if (all_recommendations_are_relevant(policy[s],s)):
            
            if np.random.uniform() < alpha:  # If all recommended items are relevant
                s_prime = int(np.random.choice(policy[s]))  # Pick a random item from relevant recommended items
            else:  # If at least one recommended item is not relevant
                s_prime = np.random.randint(K)  # Pick a random item
        else:
            s_prime = np.random.randint(K)  # Pick a random item
        
        s=s_prime
        cost_total += Cost[s]  # Add the cost of the picked item
    return cost_total

def simulation(policy):
    total_cost = 0
    num_of_episodes=50000
    for _ in range(num_of_episodes):
        total_cost  += simulate_session(policy)

    print(total_cost/num_of_episodes)

simulation(P_opt1)
#simulation(P_opt2)
simulation(pi_Q_learning)