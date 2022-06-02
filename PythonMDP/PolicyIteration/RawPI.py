#%%
import numpy as np
from random import choice
from numpy.linalg import solve
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve


class PolicyIteration(object):
    def __init__(self, states, controls, transition_matrix, expected_reward_vector, alpha=0.9, sparse_m=True):
        self.states = states
        self.controls = controls
        self.transition_matrix = transition_matrix
        self.expected_reward_vector = expected_reward_vector
        self.alpha = alpha
        self.sparse = sparse_m

    def policy_evaluation(self, policy):
        """
        Find the value function of an specific stacionary policy
        :param:
            policy: tuple with the pair state-control
        :return:
            j_mu: float, value function of the given desition rule
        """
        # Select the indexes that will define my transition matrix
        rows_index = [state + control * len(self.states) for state, control in policy]

        # Define the current transition matrix
        policy_probability_matrix = self.transition_matrix[rows_index]
        # Define the expected reward vector
        policy_reward_vector = self.expected_reward_vector[rows_index]
        # Value function of the given policy via (I - mu*P_mu)J_mu = g_mu
        if not self.sparse:
            j_mu = solve(np.identity(len(self.states)) - self.alpha * policy_probability_matrix, policy_reward_vector)
        else:
            A = identity(len(self.states)) - self.alpha * policy_probability_matrix
            j_mu = spsolve(A, policy_reward_vector)
        return j_mu

    def policy_improvement(self, policy_value):
        """
        Performs the policy improvement
        :param policy_value:
        :return:
        """
        # Separates the expected reward vectors of each control
        reshaped_rew_vector = np.split(self.expected_reward_vector, len(self.controls))
        # Separates the probabilty matrices of each control
        if not self.sparse:
            reshaped_prob_matrixes = np.split(self.transition_matrix, len(self.controls))
        else:
            reshaped_prob_matrixes = np.split(self.transition_matrix.A, len(self.controls))
        # Creates a list of all the J-values
        values = []
        for control in range(len(self.controls)):
            factor = np.matmul(reshaped_prob_matrixes[control], policy_value).reshape((len(self.states), 1))
            value = reshaped_rew_vector[control] + self.alpha * factor
            values.append(value.reshape((len(self.states), 1)))

        # take the argmin
        concatenated_values = np.concatenate(values, axis=1)
        improved_policy = np.argmin(concatenated_values, axis=-1)
        return improved_policy

    def policy_iteration(self):
        """
        Performs th ordinary Policy Iteration Algorithm
        :return:
        """
        # Empty parameters to return
        value_functions = []
        policies = []
        end = False

        # Inicialization of a random policy
        randomize_policy = [choice(list((self.controls.keys()))) for _ in range(len(self.states))]
        randomize_policy = map(int, randomize_policy)
        mu_0 = tuple(zip(self.states.keys(), randomize_policy))
        n = 0
        while not end:
            policies.append(mu_0)
            # Policy evaluation
            j_n = self.policy_evaluation(mu_0)
            value_functions.append(j_n)
            # Policy improvement
            mu_n = self.policy_improvement(policy_value=j_n)
            # Checking the convergence
            mu_n = tuple(zip(self.states.keys(), mu_n))
            if n != 0:
                last = value_functions[-1]
                last_1 = value_functions[-2]
                if np.allclose(last, last_1):
                    end = True
            # Update values
            n += 1
            mu_0 = mu_n
        return value_functions, policies, n


if __name__ == '__main__':
    from PythonMDP.Examples import MachineReplacement

    machine = MachineReplacement(machine_states=10, cost_fuction=lambda x: x, new_machine_cost=70)
    estados = machine.states
    acciones = machine.controls
    matriz_transicion = machine.prob_matrix
    recompensas = machine.reward_vector

    # Algoritmo
    policy_iterator = PolicyIteration(states=estados,
                                      controls=acciones,
                                      transition_matrix=matriz_transicion,
                                      expected_reward_vector=recompensas)

    js, politicas, iteraciones = policy_iterator.policy_iteration()
    print(politicas[-1])

