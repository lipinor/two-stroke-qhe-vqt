import pennylane as qml
import numpy as np
import itertools
import scipy
import qiskit
import matplotlib.pyplot as plt

from typing import Union
from qiskit.utils.backend_utils import is_aer_provider
from qiskit.opflow import (
    CircuitSampler,
    ExpectationFactory,
    CircuitStateFn,
    StateFn,
    I, X, Y, Z
)

from scipy.optimize import minimize

def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
        x (float): function argument.

    Returns:
        float: evaluated sigmoid function.
    """
    
    return np.exp(x) / (np.exp(x) + 1)

def prob_dist(params: list) -> np.array:
    """Calculates the probability distribution of the sigmoid function.

    Args:
        params (list): list of probability distribution parameters.

    Returns:
        np.array: return array with stacked parameters. 
    """
    return np.vstack([1 - sigmoid(params), sigmoid(params)]).T

class VQT:
    def __init__(self, 
        hamiltonian: qiskit.opflow.OperatorBase, 
        beta: float,
        ansatz: list,  
        backend: Union[qiskit.providers.BaseBackend, qiskit.utils.QuantumInstance]
    ) -> None:
        """Initalizes the class.

        Args:
            hamiltonian (qiskit.opflow.OperatorBase): Hamiltonian that you want to get 
            the expectation value.
            beta (float): inverse temperature.
            ansatz (qiskit.QuantumCircuit): List containing the QuantumCircuit that you want 
            to get the expectation value.
            backend (Union[qiskit.providers.BaseBackend, qiskit.utils.QuantumInstance]): Backend
            that you want to run.
        """

        self.hamiltonian = hamiltonian
        self.beta = beta
        self.ansatz = ansatz
        self.backend = backend

    def calculate_entropy(self, distribution: list) -> np.array:
        """Calculates the entropy for a given distribution.
        Returns an array of the entropy values of the different initial density matrices.

        Args:
            distribution (list): probability distribution

        Returns:
            np.array: entropy values of the different initial density matrices
        """
        total_entropy = 0
        for d in distribution:
            total_entropy += -1 * d[0] * np.log(d[0]) + -1 * d[1] * np.log(d[1])

        return total_entropy

    def convert_list(self, params: np.array, n_rotations: int=3) -> tuple:
        """Converts the list of parameters of the ansatz and
        split them into the distribution parameters and 
        the ansatz parameters.

        Args:
            params (np.array): list of ansatz params.
            n_rotations (int, optional): number of rotation parameters.
            per layer. Defaults to 3.

        Returns:
            tuple: tuple containing two lists, one with the distribution 
            parameters and the other with the ansatz parameters.
        """
        # Separates the list of parameters
        dist_params = params[0:self.nr_qubits]
        ansatz_params = params[self.nr_qubits:]

        return dist_params, ansatz_params

    def sample_ansatz(self,
        ansatz: qiskit.QuantumCircuit,
        ansatz_params: list, 
        ) -> float:
        """Samples a hamiltonian given an ansatz, which is a Quantum circuit
        and outputs the expected value given the hamiltonian.

        Args:
            ansatz (qiskit.QuantumCircuit): Quantum circuit that you want to get the expectation
            value.
            ansatz_params (list): List of parameters of the ansatz.

        Returns:
            float: Expectation value.
        """

        if qiskit.utils.quantum_instance.QuantumInstance == type(self.backend):
            sampler = CircuitSampler(self.backend, param_qobj=is_aer_provider(self.backend.backend))
        else:
            sampler = CircuitSampler(self.backend)

        expectation = ExpectationFactory.build(operator=self.hamiltonian, backend=self.backend)
        observable_meas = expectation.convert(StateFn(self.hamiltonian, is_measurement=True))

        ansatz = ansatz.bind_parameters(ansatz_params)

        ansatz_circuit_op = CircuitStateFn(ansatz)

        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()
        sampled_expect_op = sampler.convert(expect_op)

        return np.real(sampled_expect_op.eval())

    def exact_cost(self, params: list) -> float:
        """Calculates the exact cost of the ansatz.

        Args:
            params (list): list contaning the parameters for the 
            distribution and the ansatz.

        Returns:
            float: calculated cost function.
        """

        # Transforms the parameter list
        dist_params, ansatz_params = self.convert_list(params)

        # Creates the probability distribution
        distribution = prob_dist(dist_params)

        # Generates a list of all computational basis states of the 
        # qubit system
        combos = itertools.product([0, 1], repeat=self.nr_qubits)
        s = [list(c) for c in combos]

        # Passes each basis state through the variational circuit 
        # and multiplies the calculated energy EV with the associated 
        # probability from the distribution
        cost = 0
        for i in s:
            result = self.sample_ansatz(self.ansatz[i[0]],
                                        ansatz_params
                                       )
            for j in range(0, len(i)):
                result = result * distribution[j][i[j]]
            cost += result

        # Calculates the entropy and the final cost function
        entropy = self.calculate_entropy(distribution)
        final_cost = self.beta * cost - entropy

        return final_cost
    
    def cost_execution(self, params):
        """Executes the cost step, counts the number of iterations,
        and appends the cost history to a list.

        Args:
            params (list): list contaning the parameters for the 
            distribution and the ansatz. 

        Returns:
            float: calculated cost function.
        """
        
        cost = self.exact_cost(params)

        self.history.append(float(cost))

        if self.iterations % 5 == 0:
            print("Cost at Step {}: {}".format(self.iterations, cost))

        self.iterations += 1
        return cost

    def out_params(self, nr_qubits, depth, method='COBYLA', n_rotations: int=3, random_seed=42, 
    plot=True, plot_color='tab:blue'):
        """Performs the classical optimization of the cost function.

        Args:
            nr_qubits (int): number of qubits.
            depth (int): depth of the ansatz.
            method (str): classical optimization method. Defaults to 'COBYLA'.
            n_rotations (int, optional): number of rotations per layer. Defaults to 3.
            random_seed (int, optional): random seed for reproducibility. Defaults to 42.
            plot (bool, optional): If True, show cost plot. Defaults to True.
            plot_color (str, optional): Set the color of the plot. Defaults to 'tab:blue'.

        Returns:
            list: The parameters of the probability distribution and of the ansatz.
        """

        self.iterations = 0
        self.history = []
        
        np.random.seed(random_seed)
        
        self.nr_qubits = nr_qubits
        self.depth = depth

        number =  self.nr_qubits * (1 + self.depth * n_rotations)
        params = [np.random.randint(-300, 300) / 100 for i in range(0, number)]
        
        print("Training...")
        
        out = minimize(self.cost_execution, 
                       x0=params,
                       method=method, 
                       options={"maxiter": 20}
                       )
        
        print("Finished after " + str(self.iterations) + " steps.")

        if plot == True:
            self.plot_training(plot_color)
    
        return out["x"], self.history      

    def plot_training(self, plot_color):
        plt.plot(range(len(self.history)), self.history, '.-', color=plot_color)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
    
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