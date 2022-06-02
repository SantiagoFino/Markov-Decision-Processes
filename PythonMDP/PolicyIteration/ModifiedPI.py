# %%
import numpy as np
from numpy.random import rand


class ModifiedPolicyIteration(object):
    def __init__(self, states, controls, transition_matrix, expected_reward_vector, alpha=0.9, sparse=True, m=20):
        self.states = states
        self.controls = controls
        self.transition_matrix = transition_matrix
        self.rew_vector = expected_reward_vector
        self.alpha = alpha
        self.sparse = sparse
        self.m = m

    def policy_improvement(self, value_function):
        """
        Performs the policy improvement
            :param value_function:
            :return:
        """
        # Separates the expected reward vectors of each control
        reshaped_rew_vector = np.split(self.rew_vector, len(self.controls))
        # Separates the probabilty matrices of each control
        if not self.sparse:
            reshaped_prob_matrixes = np.split(self.transition_matrix, len(self.controls))
        else:
            reshaped_prob_matrixes = np.split(self.transition_matrix.A, len(self.controls))

        # Creates a list of all the J-values
        values = []
        for control in range(len(self.controls)):
            factor = np.matmul(reshaped_prob_matrixes[control], value_function).reshape((len(self.states), 1))
            value = reshaped_rew_vector[control] + self.alpha * factor
            values.append(value.reshape((len(self.states), 1)))

        # take the argmin
        concatenated_values = np.concatenate(values, axis=1)
        improved_policy = np.argmin(concatenated_values, axis=1)
        return improved_policy

    def aprox_policy_evaluation(self, policy, value_function):
        """
        Approximates the value vector of a given policy from a given value function
        :param policy:
        :param value_function:
        :return:
        """
        # Select the indexes that will define the transition matrix
        rows_index = [state + control * len(self.states) for state, control in policy]

        # Define the current transition matrix
        policy_probability_matrix = self.transition_matrix[rows_index]
        # Define the expected reward vector
        policy_reward_vector = self.rew_vector[rows_index]
        # Initial value function
        j_0 = value_function
        list_j = []
        for _ in range(self.m):
            factor = np.matmul(policy_probability_matrix, j_0).reshape((len(self.states), 1))
            j_k = policy_reward_vector + factor
            j_0 = j_k
            list_j.append(j_0)
        return j_0, list_j

    def mod_policy_iteration(self):
        """
        Performs the modified/optimistic Policy iteration Algorithm
        :return:
        """
        # Empty parameters to return
        value_functions = []
        policies = []
        end = False

        # Inicialization of a random value function
        j_0 = rand(len(self.states), 1)
        n = 0
        while not end:
            # Policy improvement
            mu = self.policy_improvement(value_function=j_0)
            mu = tuple(zip(self.states.keys(), mu))
            policies.append(mu)
            # Policy evaluation
            j_n, list_j = self.aprox_policy_evaluation(policy=mu, value_function=j_0)
            value_functions.append(j_n)
            # Checking the convergence
            if n != 0:
                if np.linalg.norm(value_functions[-1] - value_functions[-2]) < 0.00001 or n > 1000:
                    end = True
            # Update values
            n += 1
        return value_functions, policies, n


if __name__ == '__main__':
    from PythonMDP.Examples import EjemploNoComputacional
    ex = EjemploNoComputacional()
    estados = ex.estados
    acciones = ex.controles
    m_tr = ex.matriz_concatenada
    v_rec = ex.g_concatenado

    mod_policy_iterator = ModifiedPolicyIteration(states=estados,
                                                  controls=acciones,
                                                  transition_matrix=m_tr,
                                                  expected_reward_vector=v_rec,
                                                  sparse=False)

    v_f, politicas, iteraciones = mod_policy_iterator.mod_policy_iteration()
