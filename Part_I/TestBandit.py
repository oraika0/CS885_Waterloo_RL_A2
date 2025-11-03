import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt

def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

summary_trials = {'epsilon_greedy': [], 'thompson_sampling': [], 'UCB': []}
for i in range(1000):
    print(f"\n--- Trial {i+1} ---")
    # Test epsilon greedy strategy
    empiricalMeans, epsRewardCurve = banditProblem.epsilonGreedyBandit(nIterations=200)
    # print("\nepsilonGreedyBandit results")
    # print(empiricalMeans)
    summary_trials['epsilon_greedy'].append(epsRewardCurve)

    # Test Thompson sampling strategy
    empiricalMeans, thompsonRewardCurve = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
    # print("\nthompsonSamplingBandit results")
    # print(empiricalMeans)
    summary_trials['thompson_sampling'].append(thompsonRewardCurve)
    
    # Test UCB strategy
    empiricalMeans, UCBRewardCurve = banditProblem.UCBbandit(nIterations=200)
    # print("\nUCBbandit results")
    # print(empiricalMeans)
    summary_trials['UCB'].append(UCBRewardCurve)

curves = {k: np.asarray(v) for k, v in summary_trials.items()}
for k, v in curves.items():
    print(f"{k} shape: {v.shape}")
    assert v.shape[1] == 200, "Wrong shape"

# 逐點平均與標準差
avg = {k: v.mean(axis=0) for k, v in curves.items()}
print(avg)
std = {k: v.std(axis=0)  for k, v in curves.items()}

x = np.arange(1, 200+1)

plt.figure(figsize=(8,5))
plt.plot(x, avg['epsilon_greedy'], label='ε-greedy (ε=1/nIter)')
plt.fill_between(x, avg['epsilon_greedy']-std['epsilon_greedy'], 
                    avg['epsilon_greedy']+std['epsilon_greedy'], alpha=0.1)

plt.plot(x, avg['thompson_sampling'], label='Thompson (Beta(1,1))')
plt.fill_between(x, avg['thompson_sampling']-std['thompson_sampling'], 
                    avg['thompson_sampling']+std['thompson_sampling'], alpha=0.1)

plt.plot(x, avg['UCB'], label='UCB1')
plt.fill_between(x, avg['UCB']-std['UCB'], 
                    avg['UCB']+std['UCB'], alpha=0.1)

plt.xlabel('Iteration')
plt.ylabel('Average reward over 1000 trials')
plt.title('Bandit learning curves (mean ± std)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('bandit_1000trials.png', dpi=200)
