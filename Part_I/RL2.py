import numpy as np
import MDP

class RL2:
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

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)
        
        S = self.mdp.nStates
        A = self.mdp.nActions

        # 1) 統計表
        trans_counts = np.zeros((A, S, S), dtype=np.int64)   # N(s,a,s')
        sa_counts    = np.zeros((A, S), dtype=np.int64)      # N(s,a)
        reward_sum   = np.zeros((A, S), dtype=np.float64)    # 累計回饋

        # 2) 模型參數的即時估計（初值用題目指定的 default / initial）
        T_hat = defaultT.copy()      # shape (A,S,S)
        R_hat = initialR.copy()      # shape (A,S)

        V = np.zeros(S)
        policy = np.zeros(S, dtype=int)
        
        gamma = self.mdp.discount
        returns_per_episode = np.zeros(nEpisodes)
        for epi in range(nEpisodes):
            s = s0
            G = 0.0
            discount_pow = 1.0
            for t in range(nSteps):
                if np.random.rand() < epsilon:
                    a = np.random.randint(A)
                else:
                    a = policy[s] if sa_counts.sum() > 0 else np.random.randint(A)

                r, s2 = self.sampleRewardAndNextState(s, a)

                trans_counts[a, s, s2] += 1
                sa_counts[a, s]        += 1
                reward_sum[a, s]       += r

                G += r * discount_pow
                discount_pow *= gamma

                na = sa_counts[a, s]
                if na > 0:
                    R_hat[a, s] = reward_sum[a, s] / na
                    T_hat[a, s, :] = trans_counts[a, s, :] / na

                # run MDP with statistics' T、R
                mdp_hat = MDP.MDP(T_hat, R_hat, self.mdp.discount)
                V, _, _ = mdp_hat.valueIteration(initialV=V, tolerance=1e-2) 
                policy = mdp_hat.extractPolicy(V)

                s = s2
            returns_per_episode[epi] = G
            
        return [V,policy, returns_per_episode]    

    def qLearning(self, s0, initialQ, nEpisodes, nSteps,
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
        
    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        used_counts = np.zeros(self.mdp.nActions)
        reward_curve = []
        for step in range(nIterations):
            # epsilon greedy
            epsilon = 1.0 / (step+1)
            
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, self.mdp.nActions)
            else:
                # action = np.random.randint(0, self.mdp.nActions)
                action = np.argmax(empiricalMeans)
            reward = self.sampleReward(self.mdp.R[action, 0])
            reward_curve.append(reward)
            used_counts[action] += 1.0
            # print(action,reward) 

            # update empirical mean
            empiricalMeans[action] = (
                empiricalMeans[action] * ((used_counts[action]-1) / used_counts[action]) 
                + reward / used_counts[action]
            )

        
        

        return empiricalMeans, reward_curve

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm 
                (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        # empiricalMeans = np.zeros(self.mdp.nActions)
        
        nArms = self.mdp.nActions 
        s = 0  # bandit 當作單一 state 0

        # copy 先驗到可變參數
        alpha = prior[:,0].astype(float).copy()
        beta  = prior[:,1].astype(float).copy()

        # 統計用來算最後的 empirical mean
        counts = np.zeros(nArms)
        total_reward = np.zeros(nArms)
        reward_curve = []

        for t in range(nIterations):
            # 1. 對每支手臂 sample 一個臨時的 "我覺得你的成功機率"
            sampled_means = np.zeros(nArms)
            for a in range(nArms):
                draws = np.random.beta(alpha[a], beta[a], size=k)  # 從 Beta 抽樣
                sampled_means[a] = np.mean(draws)  # k>1 就是平均一下，讓它不那麼抖

            # 2. 選擇抽樣結果最高的 arm
            chosen_arm = np.argmax(sampled_means)

            # 3. 執行那個動作，拿 reward (0 or 1)
            reward = self.sampleReward(self.mdp.R[chosen_arm, s])
            reward_curve.append(reward)
            # 4. 更新那支 arm 的後驗參數 alpha/beta
            if reward > 0:
                alpha[chosen_arm] += 1.0  # 成功
            else:
                beta[chosen_arm]  += 1.0  # 失敗

            # 5. 更新經驗統計
            counts[chosen_arm] += 1.0
            total_reward[chosen_arm] += reward

        # 最後回傳每支 arm 的經驗平均報酬
        empiricalMeans = np.zeros(nArms)
        pulled = counts > 0
        empiricalMeans[pulled] = total_reward[pulled] / counts[pulled]

        return empiricalMeans, reward_curve

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        # empiricalMeans = np.zeros(self.mdp.nActions)
        nArms = self.mdp.nActions 
        s = 0 

        # 統計用來算最後的 empirical mean
        counts = np.zeros(nArms)
        total_reward = np.zeros(nArms)
        reward_curve = []
        
        for i in range(nArms):
            reward = self.sampleReward(self.mdp.R[i, s])
            reward_curve.append(reward)
            counts[i] += 1
            total_reward[i] += reward

        for t in range(nArms, nIterations):
            UCB_values = np.zeros(nArms)
            for a in range(nArms):
                UCB_values[a] = total_reward[a] / counts[a] + np.sqrt(2 * np.log(t) / counts[a])
                
            # 2. 選擇抽樣結果最高的 arm
            chosen_arm = np.argmax(UCB_values)

            # 3. 執行那個動作，拿 reward (0 or 1)
            reward = self.sampleReward(self.mdp.R[chosen_arm, s])
            reward_curve.append(reward)
            counts[chosen_arm] += 1
            total_reward[chosen_arm] += reward

        # 最後回傳每支 arm 的經驗平均報酬
        empiricalMeans = np.zeros(nArms)
        empiricalMeans = total_reward / counts

        return empiricalMeans, reward_curve