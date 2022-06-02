import numpy as np
from numpy.random import zipf
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix


class EjemploNoComputacional(object):
    def __init__(self):
        self.estados = {0: '1', 1: '2'}
        self.controles = {0.0: 'u_1', 1.0: 'u_2'}

        # Matrices de transicion
        self.matriz_u1 = np.array([3 / 4, 1 / 4, 3 / 4, 1 / 4]).reshape((2, 2))
        self.matriz_u2 = np.array([1 / 4, 3 / 4, 1 / 4, 3 / 4]).reshape((2, 2))
        self.matriz_concatenada = np.concatenate([self.matriz_u1, self.matriz_u2], axis=0)

        # Vectores de recompensa
        self.g_1 = np.array([2, 0.5]).reshape((2, 1))
        self.g_2 = np.array([1, 3]).reshape((2, 1))
        self.g_concatenado = np.concatenate([self.g_1, self.g_2], axis=0)


class Parking(object):
    """
    Parcking example which is distribute following a zipf law
    """

    def __init__(self, parking_size):
        self.parking_size = parking_size
        nombres_parqueaderos = ['Parking #' + str(i) for i in range(self.parking_size)]
        self.states = dict(zip(np.arange(self.parking_size), nombres_parqueaderos))
        self.controls = {0.0: 'No park', 1.0: 'Park'}

        # Probabilities of each slot
        zipf_p = zipf(a=4, size=self.parking_size)[::-1]
        self.probabilities = np.sort(zipf_p / np.sum(zipf_p))
        self.concatenated_matrix = self.probability_matrix()

        # Reward vectors
        self.no_park_rewards = np.hstack([np.zeros(self.parking_size - 1),
                                          float(- self.parking_size)]).reshape((self.parking_size, 1))
        self.park_rewards = 10 / np.arange(start=1, stop=self.parking_size + 1)[::-1].reshape((self.parking_size, 1))
        self.reward_vector = np.concatenate([self.no_park_rewards, self.park_rewards], axis=1)

    def probability_matrix(self):
        """
        Constructs the transition matrix of each control
        :return:
        """
        # No park - matrix
        data = np.ones(self.parking_size)
        no_park_matrix = diags(data, shape=(self.parking_size, self.parking_size))
        # Park - matrix
        park_matrix = diags(self.probabilities, shape=(self.parking_size, self.parking_size))
        concatenated_matrix = sparse.vstack([no_park_matrix, park_matrix])
        return concatenated_matrix


class MachineReplacement(object):
    def __init__(self, machine_states=100, cost_fuction=lambda i: i ** (3/2), new_machine_cost=500, zipf_p=1.5):
        """
        Constructor
        :param machine_states:
        :param cost_fuction:
        :param new_machine_cost
        """
        self.machine_states = machine_states
        self.cost_fuction = cost_fuction
        self.new_machine_cost = new_machine_cost
        self.zipf_p = zipf_p
        performance = ['Performance: ' + str(np.round(perc, 2)) +
                       '%' for perc in np.linspace(start=0,
                                                   stop=100,
                                                   num=self.machine_states)[::-1]]
        self.performances = np.linspace(start=0, stop=100, num=self.machine_states)
        self.states = dict(zip(np.arange(self.machine_states), performance))
        self.controls = {0.0: 'Keep the machine', 1.0: 'New machine'}
        self.prob_matrix = self.probabilities()
        self.reward_vector = self.costs()

    def probabilities(self):
        # Keep-using probabilities
        distributions = []
        for size in range(self.machine_states-1):
            row = np.sort(zipf(a=self.zipf_p, size=self.machine_states - size-1))[::-1]
            distributions.append(row / np.sum(row))

        used_machine_probs = np.zeros((self.machine_states, self.machine_states))
        for column in range(self.machine_states-1):
            used_machine_probs[column, column+1:] = distributions[column]
        # When the machine arrives to an 0%-performance state, it'll stay there
        used_machine_probs[self.machine_states-1, self.machine_states-1] = 1.0
        used_machine_probs = sparse.csr_matrix(used_machine_probs)

        # New machine probabilities
        one_colums = np.ones(self.machine_states).reshape((self.machine_states, 1))
        sparse_m = csc_matrix((self.machine_states, self.machine_states - 1))
        new_m_probabilities = sparse.hstack([one_colums, sparse_m])

        # Final matrix
        concatenated_matrix = csr_matrix(sparse.vstack([used_machine_probs, new_m_probabilities]))
        return concatenated_matrix

    def costs(self):
        # Ask for a new machine costs
        cost_new = np.ones((self.machine_states, 1))*self.new_machine_cost

        # Cost
        cost_per_use = self.cost_fuction(self.performances.reshape((self.machine_states, 1)))
        concatenated_cost = np.vstack([cost_per_use, cost_new])
        return concatenated_cost


if __name__ == '__main__':
    machine = MachineReplacement(machine_states=10)
    print(f'Estados: {machine.states}, Controles: {machine.controls}')
    print(f'Vector Recompensas: {machine.reward_vector}')
    print(f'Matriz Transición: {machine.prob_matrix}')
