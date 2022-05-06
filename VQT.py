import pennylane as qml
import numpy as np
import itertools
import scipy

from scipy.optimize import minimize

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)

def prob_dist(params):
    """ Calculates the probability distribution of the sigmoid
    """
    return np.vstack([1 - sigmoid(params), sigmoid(params)]).T

class VQT:
    def __init__(self, quantum_circuit, dev):
        self.qnode = qml.QNode(quantum_circuit, dev)

    def calculate_entropy(self, distribution):
        """ Calculates the entropy for a given distribution.
        Returns an array of the entropy values of the different initial density matrices.
        """
        total_entropy = 0
        for d in distribution:
            total_entropy += -1 * d[0] * np.log(d[0]) + -1 * d[1] * np.log(d[1])

        return total_entropy

    def convert_list(self, params: np.array, n_rotations: int=3):
        """ Converts the list of parameters of the ansatz and
        split them into the distribution parameters and 
        the ansatz parameters.

        """
        # Separates the list of parameters
        dist_params = params[0:self.nr_qubits]
        ansatz_params = params[self.nr_qubits:]

        # Partitions the parameters into multiple lists
        split = np.split(ansatz_params, self.depth)
        rotation = []
        for s in split:
            rotation.append(np.split(s, n_rotations))

        ansatz_params = rotation

        return dist_params, ansatz_params

    def exact_cost(self, params, ham_matrix, beta):
        """ Calculates the exact cost of the ansatz.
        """

        # Transforms the parameter list
        dist_params, ansatz_params = self.convert_list(params)

        # Creates the probability distribution
        distribution = prob_dist(dist_params)

        # Generates a list of all computational basis states of our qubit system
        combos = itertools.product([0, 1], repeat=self.nr_qubits)
        s = [list(c) for c in combos]

        # Passes each basis state through the variational circuit and multiplies
        # the calculated energy EV with the associated probability from the distribution
        cost = 0
        for i in s:
            result = self.qnode(ansatz_params, ham_matrix, sample=[i[0]])
            for j in range(0, len(i)):
                result = result * distribution[j][i[j]]
            cost += result

        # Calculates the entropy and the final cost function
        entropy = self.calculate_entropy(distribution)
        final_cost = beta * cost - entropy

        return final_cost
    
    def cost_execution(self, params, ham_matrix, beta, history):
        """ executes the cost step.
        """
        cost = self.exact_cost(params, ham_matrix, beta)

        history.append(float(cost))

        if self.iterations % 5 == 0:
            print("Cost at Step {}: {}".format(self.iterations, cost))

        self.iterations += 1
        return cost

    def out_params(self, ham_matrix, beta, history, nr_qubits, depth, n_rotations: int=3, random_seed=42):
        self.iterations = 0
        
        np.random.seed(random_seed)
        
        self.nr_qubits = nr_qubits
        self.depth = depth

        number =  self.nr_qubits * (1 + self.depth * n_rotations)
        params = [np.random.randint(-300, 300) / 100 for i in range(0, number)]
        
        print("Training...")
        
        out = minimize(self.cost_execution, 
                        x0=params, 
                        args=(ham_matrix, beta, history), 
                        method="COBYLA", 
                        options={"maxiter": 20}
                        )
        
        print("Finished after " + str(self.iterations) + " steps.")
        return out["x"]        
    
    def prepare_state(self, params, hamiltonian: np.array, device) -> np.array:

        # Initializes the density matrix

        final_density_matrix = np.zeros((2 ** self.nr_qubits, 2 ** self.nr_qubits))

        # Prepares the optimal parameters, creates the distribution and the bitstrings
        dist_params, ansatz_params = self.convert_list(params)

        distribution = prob_dist(dist_params)

        combos = itertools.product([0, 1], repeat = self.nr_qubits)
        s = [list(c) for c in combos]

        # Runs the circuit in the case of the optimal parameters, for each bitstring,
        # and adds the result to the final density matrix

        for i in s:
            self.qnode(ansatz_params, hamiltonian, sample=[i[0]])
            state = device.state
            for j in range(0, len(i)):
                state = np.sqrt(distribution[j][i[j]]) * state
            final_density_matrix = np.add(final_density_matrix, np.outer(state, np.conj(state)))

        return final_density_matrix

def create_target(beta: float, hamiltonian: np.array) -> np.array:
    """Calculates the matrix form of the density matrix, by taking
     the exponential of the Hamiltonian

    """

    y = -1 * float(beta) * hamiltonian
    new_matrix = scipy.linalg.expm(np.array(y))
    norm = np.trace(new_matrix)
    final_target = (1 / norm) * new_matrix

    return final_target