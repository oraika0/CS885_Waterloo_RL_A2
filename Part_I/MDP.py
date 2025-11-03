import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # temporary values to ensure that the code compiles until this
        # function is coded
        V = initialV.copy()
        iterId = 0
        epsilon = np.inf
        
        while epsilon > tolerance :
            V_prev = V.copy()
            for s in range(self.nStates):
                Q_values = np.zeros(self.nActions)
                for a in range(self.nActions):
                    Q_values[a] = self.R[a ,s] + self.discount * np.sum(self.T[a][s, :] * V_prev)
                V[s] = np.max(Q_values)
            epsilon = np.max(np.abs(V - V_prev))
            iterId += 1
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates, dtype=int)

        for s in range(self.nStates):
            Q_values = np.zeros(self.nActions)
            for a in range(self.nActions):
                Q_values[a] = self.R[a ,s] + self.discount * np.sum(self.T[a][s, :] * V)
            policy[s] = np.argmax(Q_values)   # 選出最優 action 的 index

        return policy


    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        
        tolerance = 0.01
        it = 0
        delta = np.inf

        while delta > tolerance:
            V_prev = V.copy()

            for s in range(self.nStates):
                a = policy[s]
                
                reward = self.R[a, s]
                # γ * Σ_{s'} T(s'|s,a) * V_prev[s']
                V[s] = reward + self.discount * np.sum(self.T[a, s, :] * V_prev)

            delta = np.max(np.abs(V - V_prev))
            it += 1

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = initialPolicy.copy()
        iterId = 0

        while iterId < nIterations:
            # PE
            V = self.evaluatePolicy(policy)
            # PI
            newPolicy = self.extractPolicy(V)

            iterId += 1

            if np.array_equal(newPolicy, policy):
                break

            policy = newPolicy


        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = initialV.copy()
        iterId = 0
        epsilon = np.inf

        while iterId < nIterations and epsilon > tolerance:
            V_prev = V.copy()

            for s in range(self.nStates):
                a = policy[s]
                reward = self.R[a, s]
                future = self.discount * np.sum(self.T[a, s, :] * V_prev)
                V[s] = reward + future

            epsilon = np.max(np.abs(V - V_prev))
            iterId += 1

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = initialPolicy.copy()
        V = initialV.copy()

        iterId = 0
        epsilon = np.inf

        while iterId < nIterations and epsilon > tolerance:
            # ----- 1. Partial policy evaluation on current policy -----
            V_eval, _, _ = self.evaluatePolicyPartially(
                policy,
                V,
                nIterations=nEvalIterations,
                tolerance=1e-12
            )

            newPolicy = self.extractPolicy(V_eval)

            newV = np.zeros(self.nStates)
            for s in range(self.nStates):
                Q_values = np.zeros(self.nActions)
                for a in range(self.nActions):
                    Q_values[a] = self.R[a, s] + self.discount * np.sum(self.T[a, s, :] * V_eval)
                newV[s] = np.max(Q_values)

            epsilon = np.max(np.abs(newV - V))
            iterId += 1

            V = newV
            policy = newPolicy

        return [policy,V,iterId,epsilon]
        