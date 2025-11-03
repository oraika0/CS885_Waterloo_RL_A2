import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        # Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
        # policy = np.zeros(self.mdp.nStates,int)

        Q = initialQ.copy().astype(float)

        # visit counter for learning rate schedule alpha = 1/N(s,a)
        N_sa = np.zeros_like(Q, dtype=float)

        def chooseAction(state):
            # 1) ε part: with prob epsilon, pick random action
            if np.random.rand() < epsilon:
                return np.random.randint(self.mdp.nActions)

            # 2) otherwise do Boltzmann / greedy
            q_vals = Q[:, state]

            # temperature == 0 => greedy argmax with random tie-break
            if temperature == 0:
                max_val = np.max(q_vals)
                best_as = np.where(q_vals == max_val)[0]
                return np.random.choice(best_as)

            # temperature > 0 => softmax over Q / temperature
            prefs = q_vals / float(temperature)
            # subtract max for numerical stability
            prefs = prefs - np.max(prefs)
            exp_prefs = np.exp(prefs)
            probs = exp_prefs / np.sum(exp_prefs)
            return np.random.choice(np.arange(self.mdp.nActions), p=probs)

        gamma = self.mdp.discount

        for _ in range(nEpisodes):
            state = s0
            for _step in range(nSteps):

                # pick action using exploration strategy
                action = chooseAction(state)

                # interact with env: sample reward and next state
                reward, nextState = self.sampleRewardAndNextState(state, action)

                # compute TD target: r + gamma * max_a' Q(nextState,a')
                best_next = np.max(Q[:, nextState])
                td_target = reward + gamma * best_next

                # learning rate alpha = 1 / N(s,a) (after increment)
                N_sa[action, state] += 1.0
                alpha = 1.0 / N_sa[action, state]

                # Q-learning update
                Q[action, state] = Q[action, state] + alpha * (td_target - Q[action, state])

                # move on
                state = nextState

        # derive final greedy policy from Q
        policy = np.zeros(self.mdp.nStates, dtype=int)
        for s in range(self.mdp.nStates):
            q_vals = Q[:, s]
            max_val = np.max(q_vals)
            best_as = np.where(q_vals == max_val)[0]
            policy[s] = np.random.choice(best_as)

        return [Q, policy]
    
    def qLearning_with_returns(self, s0, initialQ, nEpisodes, nSteps,
                            epsilon=0, temperature=0):
        Q = initialQ.copy().astype(float)
        N_sa = np.zeros_like(Q, dtype=float)
        
        def chooseAction(state):
            # epsilon-greedy outer layer
            if np.random.rand() < epsilon:
                return np.random.randint(self.mdp.nActions)
            q_vals = Q[:, state]
            # if temperature==0 -> greedy
            if temperature == 0:
                max_val = np.max(q_vals)
                best_as = np.where(q_vals == max_val)[0]
                return np.random.choice(best_as)
            # Boltzmann softmax
            prefs = q_vals / float(temperature)
            prefs = prefs - np.max(prefs)
            exp_prefs = np.exp(prefs)
            probs = exp_prefs / np.sum(exp_prefs)
            return np.random.choice(np.arange(self.mdp.nActions), p=probs)
        
        gamma = self.mdp.discount
        returns_per_episode = np.zeros(nEpisodes)
        
        for ep in range(nEpisodes):
            state = s0
            G = 0.0              # cumulative discounted reward
            discount_pow = 1.0   # will track gamma^t
            for step in range(nSteps):
                # pick action
                action = chooseAction(state)
                # env step
                reward, nextState = self.sampleRewardAndNextState(state, action)
                # TD target
                best_next = np.max(Q[:, nextState])
                td_target = reward + gamma * best_next
                # learning rate alpha = 1/N(s,a)
                N_sa[action, state] += 1.0
                alpha = 1.0 / N_sa[action, state]
                # Q update
                Q[action, state] = Q[action, state] + alpha * (td_target - Q[action, state])
                # accumulate discounted reward for plotting
                G += discount_pow * reward
                discount_pow *= gamma
                # move forward
                state = nextState
            returns_per_episode[ep] = G
        # final greedy policy
        policy = np.zeros(self.mdp.nStates, dtype=int)
        for s in range(self.mdp.nStates):
            q_vals = Q[:, s]
            max_val = np.max(q_vals)
            best_as = np.where(q_vals == max_val)[0]
            policy[s] = np.random.choice(best_as)
        return [Q, policy, returns_per_episode]
