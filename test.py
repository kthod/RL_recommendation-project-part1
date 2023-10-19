import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt 
# Environment parameters
K = 5  # Number of content items
u_min = 0.5  # Threshold for content relevance
C = int(0.2 * K)  # Number of cached items
#C = 2
# User model parameters
N = 1  # Number of recommended items
q = 0.2  # Probability of ending the viewing session
alpha = 0.9  # Probability of selecting a recommended item

# Generate random relevance values
U = np.random.rand(K, K)
np.fill_diagonal(U, 0)  # Set diagonal elements to 0
# U = np.array([[0., 0.09709217, 0.95697935, 0.76421269, 0.79379138],
#               [0.85679266, 0., 0.73115609, 0.97025111, 0.00706508],
#               [0.38327773, 0.27582305, 0., 0.40938946, 0.70918518],
#               [0.27415892, 0.89691232, 0.47103534, 0., 0.97776446],
#               [0.06699551, 0.96500574, 0.00547615, 0.74654658, 0.]])
U = np.array([[0.0, 0.8, 0.3, 0.6, 0.1],
              [0.8, 0.0, 0.7, 0.4, 0.3],
              [0.3, 0.7, 0.0, 0.2, 0.9],
              [0.6, 0.4, 0.2, 0.0, 0.8],
              [0.1, 0.3, 0.9, 0.8, 0.0]])
Cost = [1]*(K-C) +[0]*C
Cost = [1,0,1,0,1]
#random.shuffle(Cost)
print(Cost)


# The next few lines are mostly for accounting
Tmax = 10000
# size = len(P)
# n = m = np.sqrt(size)
# print(size)
# Vplot = np.zeros((size,Tmax)) #these keep track how the Value function evolves, to be used in the GUI
# Pplot = np.zeros((size,Tmax)) #these keep track how the Policy evolves, to be used in the GUI
t = 0



def define_rewards():
    Rewards = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            Rewards[i][j] = 1*U[i][j] - 1*( Cost[j] )
    return Rewards

Rewards =define_rewards()
#this one is generic to be applied in many AI gym compliant environments
def all_recommendations_are_relevant(recommendations,s):
    for u in recommendations:
        if U[s][int(u)]<u_min:
            return False
    return True
    
def policy_evaluation(recommendations, U, gamma = 1.0, epsilon = 1e-10):  #inputs: (1) policy to be evaluated, (2) model of the environment (transition probabilities, etc., see previous cell), (3) discount factor (with default = 1), (4) convergence error (default = 10^{-10})
    t = 0   #there's more elegant ways to do this
    prev_V = np.zeros(K) # use as "cost-to-go", i.e. for V(s')
    while True:
        # print(t)
        V = np.zeros(K) # current value function to be learnerd
        Q = np.zeros((K,K))
        for s in range(K):  # do for every state
            if (all_recommendations_are_relevant(recommendations[s],s)):
                for s_prime in range(K):  # do for every state:  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                    # print(s_prime)
                    # print(recommendations[s])
                    if s_prime in recommendations[s]:
                        V[s] +=(alpha/N + (1-alpha)/K)* (Rewards[s][s_prime] + gamma * prev_V[s_prime])
                    else:
                        V[s] += ((1-alpha)/K) * (Rewards[s][s_prime] + gamma * prev_V[s_prime])
                    Q[s][s_prime] = Rewards[s][s_prime] + gamma * prev_V[s_prime]
            else:
                for s_prime in range(K):  # do for every state:  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                    V[s] += (1/K) * (Rewards[s][s_prime] + gamma * prev_V[s_prime])
                    Q[s][s_prime] = Rewards[s][s_prime] + gamma * prev_V[s_prime]

        if np.max(np.abs(prev_V - V)) < epsilon or t>Tmax: #check if the new V estimate is close enough to the previous one; 
            break # if yes, finish loop
        prev_V = V.copy() #freeze the new values (to be used as the next V(s'))
        t += 1
        # Vplot[:,t] = prev_V  # accounting for GUI  
    return 1/t*V,Q

def get_N_max_values(Q):
    sorted_Q = sorted(range(len(Q)), key=lambda x: Q[x], reverse=True)

    return sorted_Q[:N]


def policy_improvement(Q, U, gamma=1.0):  # takes a value function (as the cost to go V(s')), a model, and a discount parameter
    new_pi = np.zeros((K,N))
    #new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]  # this basically creates the new (improved) policy by choosing at each state s the action a that has the highest Q value (based on the Q array we just calculated)
    # lambda is a "fancy" way of creating a function without formally defining it (e.g. simply to return, as here...or to use internally in another function)
    # you can implement this in a much simpler way, by using just a few more lines of code -- if this command is not clear, I suggest to try coding this yourself
    for s in range(K):
        Q[s][s] = float('-inf')
        new_pi[s] = get_N_max_values(Q[s])

    return new_pi

# policy iteration is simple, it will call alternatively policy evaluation then policy improvement, till the policy converges.

def policy_iteration(U, gamma = 1.0, epsilon = 1e-10):
    t = 0
    # random_actions = np.random.choice(tuple(P[0].keys()), len(P))     # start with random actions for each state  
    # pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]     # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)
    
    pi = np.random.randint(0, K, size=(K, N))  # Random initial policy
    V = np.zeros(K)
    Q = np.zeros((K,K))
    Vavg = []
    while True:
        # print(t)
        Vavg.append(np.average(V))
        old_pi = pi.copy()  #keep the old policy to compare with new
        V,Q = policy_evaluation(pi,U,gamma,epsilon)   #evaluate latest policy --> you receive its converged value function
        pi = policy_improvement(Q,U,gamma)          #get a better policy using the value function of the previous one just calculated 
        
        t += 1
        # Pplot[:,t]= [pi(s) for s in range(len(P))]  #keep track of the policy evolution
        # Vplot[:,t] = V                              #and the value function evolution (for the GUI)
        if (t==20):
            t=20
        if compare_two_policies(pi,old_pi) and t>0: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
            break
    print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
   
    #print(Vavg)
    plt.plot(np.arange(1,t+1),Vavg , label = "UCB") 

    return V,pi
    
def compare_two_policies(pi,old_pi):

    for k in range(K):
        for n in range(N):
            if (pi[k][n]!=old_pi[k][n]):
                return False

    return True

def Q_learning(U,gamma,delta, epsilon,learning_rate):
    Q = np.zeros((K,K))
    prev_Q = np.zeros((K,K))
    t = 0
    while True:
        s = np.random.randint(K)
        while True:
            if np.random.random() < epsilon:  # Explore if e(t) times
                s_prime = np.random.randint(K)
            else:  # Exploit 1-e(t) times
                s_prime = np.argmax(Q[s])
            if np.random.random() < q:
                target = Rewards[s][s_prime]
                Q[s][s_prime] = (1-learning_rate)*prev_Q[s][s_prime] + learning_rate*target
                break
            else:
                target = Rewards[s][s_prime]+gamma*np.max(prev_Q[s_prime])
            Q[s][s_prime] = (1-learning_rate)*prev_Q[s][s_prime] + learning_rate*target
            s=s_prime
        t+=1
        epsilon = (t+1)**(-1/3)*(K*math.log(t+1))**(1/3)
        #learning_rate = learning_rate*1/t
        if np.max(np.abs(prev_Q - Q)) < delta or t>K*10000 : #check if the new V estimate is close enough to the previous one;
            break # if yes, finish loop
        prev_Q = Q.copy()
    print(t)
    return Q

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
            
        
        #recommended_items = policy[s]  # Get recommended items
        #relevant_recommended_items = [item for item in recommended_items if relevance[current_item][item] == 1]  # Filter out non-relevant items

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
    num_of_episodes=100000
    for _ in range(num_of_episodes):
        total_cost  += simulate_session(policy)

    print(total_cost/num_of_episodes)



pi_Q_learning  = np.zeros((K,N))

Q = Q_learning(U,1-q,0.001,0.1,0.1)

for s in range(K):
    #Q[s][s] = float('-inf')
    pi_Q_learning[s] = get_N_max_values(Q[s])

print(pi_Q_learning)

V_opt,P_opt = policy_iteration(U,1-q,0.001)   #just example of calling the various new functions we created.

print("SIMULATION")
simulation(P_opt)

print("SIMULATION")
simulation(pi_Q_learning)

chat_pi = np.array([[1],
 [3],
 [1],
 [1],
 [3]])
print("SIMULATION")
simulation(chat_pi)
# V_pi,Q = policy_evaluation(P_opt,U,1-q,0.001)
# V_q,Q = policy_evaluation(pi_Q_learning,U,1-q,0.001)
# #V_chat,Q = policy_evaluation(chat_pi,U,1-q,0.001)
# print(np.average(V_pi))
# print(np.average(V_q))
#print(np.average(V_chat))
print(U)
print("#############################")
#print(V_opt)
print("#############################")
print(P_opt)

plt.show()
