import pennylane as qml 
from pennylane import numpy as np 
import time 
from itertools import product
import cvxpy as cp
import time
import queue 

class Solver: 
    

    def __init__ (self, dev : qml.device, ansatz : str = 'CIA', gate : str = 'Z', preprocessing : str = '', postprocessing : str = '', iterations_limit : int = 700, alpha : float = 0.1, accepterr : list[float] = [0.1], baren_check : bool = False, baren_threshold : int = 10, baren_rerr: float = 0.1): 
        
        '''
        Circuits and optimizers class

        # Ansatzes list
            classical-inspired **[CIA]**
            multi-angle-hardware-efficient **[MA-HEA]**
            hardware-efficient **[HEA]**
            hardware-efficient-alt **[ALT]**
            problems-inspired **[PIA]**

            
        Parameters
        ----------
        dev : qml.device 
            On which quantum device problems should be solved 
        ketdev : qml.device 
            On which qunatum device ket vector should me measured
        ansatz : str
            an ansatz to be used 
        preprocessing : str 
            'WS' or ''
        postprocessing : str
            'CVaR' or ''
        iterations_limit : int 
            limit of iterations implemented
        alpha : float 
            CVaR coefficient
        accepterr : list[float] 
            list of errors with which we conclude fitted solution to be correct. Must be increasing
        baren_check : bool = False 
            whether baren plateau 
        baren_threshold : int = 10
            how many gradients should be calculated for baren plateau check
        baren_rerr : float 
            relative error for baren plateau identification 
        '''
        self.dev = dev 
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.ansatz = ansatz 
        self.iteration_limit = iterations_limit 
        self.accepterr = accepterr 
        self.alpha = alpha
        self.baren_check = baren_check 
        self.baren_threshold = baren_threshold
        self.baren_rerr = baren_rerr 
        self.gate = gate

    def _calculateQUBO (self, Q : np.ndarray, state_vector : np.ndarray):
        '''Calcs QUBO form's value by given [0,1] bitvector

        Parameters
        ----------
        Q : np.ndarray 
            QUBO form 
        state_vector : np.ndarray 
            bitvector
        Returns
        -------
        val : float
            QUBO form's value
        '''
        val = 0.5 * state_vector.T @ Q @ state_vector 
        return val

    def _calculateIsing (self, J : np.ndarray, h: np.ndarray, state_vector : np.ndarray):
        '''Calcs Ising form's value by given [-1,1] string 

        Parameters
        ----------
        J : np.ndarray 
            Ising form 
        h : np.ndarray
            Ising vector
        state_vector : np.ndarray 
            vector of +1 and -1 coresponding to the quantum state

        Returns
        -------
        val : float
            Ising form's value
        '''
        val = -0.5 * state_vector.T @ J @ state_vector - h.T @ state_vector
        return val
    
    def _calculateMaxcut (self, W : np.ndarray, cut_vector : np.ndarray, needs0 : bool = True):
        '''Calcs MaxCut cut value with given state_vector, s_0 = +1 

        Parameters
        ----------
        W : np.ndarray 
            weight matrix 
        cut_vector : np.ndarray 
            vector of +1 and -1 coresponding to the cut

        Returns
        -------
        cutval : float
            Maxcut value
        '''
        
        cutval = 0
        if (needs0):
            s = np.concatenate((np.array([1]), cut_vector))
        else: 
            s = cut_vector
        for i in range (0, W.shape[0]):
            for j in range (i + 1, W.shape[1]): 
                if (s[i] * s[j] == -1):
                    cutval += W[i,j]
        return cutval
    
        
    def _CVaR_expectation (self, J : np.ndarray, h : np.ndarray, samples : np.ndarray, alpha : float = 0.1): 

        ''' 
        Calculate CVaR_alpha expectation by the set of samples 

        Parameters 
        ----------  
        J : np.ndarray 
            Ising matrix 
        h : np.ndarray
            Ising vector 
        samples : np.ndarray 
            a numpy array of samples
        alpha : float 
            CVaR coefficient 

        Returns 
        -------
        expectation : float
            CVaR_alpha expectation
        '''
        
        samples = samples.T
        sampled_energies = np.sort(np.array([self._calculateIsing(J, h, sample) for sample in samples]))
        #print(sampled_energies)
        expectation = np.mean(sampled_energies[:int(alpha * sampled_energies.shape[0])])
        
        return expectation
        
    def _classical_inspired_circuit (self, J : np.ndarray, h : np.ndarray, H_cost : np.tensor, params : np.ndarray, start: str = '', initial_angles : np.ndarray = [], mode : str = 'state'):

        '''
        Classical-inspired ansatz with only 1-qubit rotations
        
        **Initial state:** put all qubits in $\ket{+}$ (for Z basis) or do nothing (for X basis)
        
        **Mixer layer:** implement RY (for Z and X basis) rotations on each qubit 
        
        **Cost layer:** do nothing

        Parameters 
        ---------
        J : np.ndarray 
            Ising matrix 
        h : np.ndarray
            Ising vector 
        H_cost : np.tensor 
            Cost hamiltonian 
        params : np.ndarray 
            - Cold : array of optimizable parameters with shape([depth, dim]) 
            - Warm : array of optimizable parameters with shape([depth, dim])
        start : str 
            cold or warm
        initital_angles : np.ndarray 
            a vector of initital angles for warm-start
        mode : str 
            state or expectation or samples
        '''

        dim = J.shape[0]

        if (self.gate == 'X'):
            # initial state - ket |0> on all qubits 
            for mixer in params:
                for i in range (dim): #mixer layer 
                    qml.RY(mixer[i], wires = i)

            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliX(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliX(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)
        
        if (self.gate == 'Z'):
            #initial state - ket |+> on all qubits

            if (start == ''):
                for i in range (dim): #cold initial state
                    qml.H(i)
            elif (start == 'WS'): 
                for i in range (dim): #warm initital state
                    qml.RY(phi = initial_angles[i], wires = i)
                
            for mixer in params:
                if (start == ''):
                    for i in range (dim): #cold mixer layer 
                        qml.RY(phi = mixer[i], wires = i)
                elif (start == 'WS'): 
                    for i in range (0, 2 * dim, 2): #warm mixer layer
                        qml.RY(phi = -mixer[i], wires = i // 2)
                        qml.RZ(phi = -2 * mixer[i + 1], wires = i // 2)
                        qml.RY(phi = 2 * mixer[i], wires = i // 2)


            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliZ(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliZ(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)

    def _problem_inspired_circuit (self, J : np.ndarray, h : np.ndarray, H_cost : np.tensor, params : np.ndarray, mode : str = 'state'):
        
        '''
        Problem-inspired ansatz circuit
        
        **Initial state:** put all qubits in $\ket{+}$ using H gate (for Z basis) or do nothing (for X basis) 
        
        **Mixer layer:** implement RX/RZ rotations on each qubit 
        
        **Cost layer:** implement Hamiltonian of XX or ZZ rotations depending on gate parameter and add some X or Z respectively
        
        Parameters 
        ---------
        J : np.ndarray 
            Ising matrix 
        h : np.ndarray
            Ising vector 
        H_cost : np.tensor 
            Cost hamiltonian 
        params : np.ndarray 
            array of optimizable parameters with shape([1, 2 * depth])
        mode : str 
            state or expectation 
        gate : str 
            Z or X depending on entaglment gate ZZ or XX respectively 

        '''
        dim = J.shape[0]
        
        if (self.gate == 'X'): 
            for layer_index in range (params.shape[0]): 
                if (layer_index % 2 == 0): #mixer layer
                    beta = params[layer_index]
                    for i in range(dim):
                        qml.RZ(phi = 2 * beta, wires = i)
                else: # cost layer 
                    alpha = params[layer_index]
                    for i in range (dim):
                        for j in range (i + 1, dim):
                            qml.IsingXX(phi = -2 * alpha * J[i,j], wires = [i, j]) 
                    for i in range (dim): 
                        qml.RX(phi = -2 * alpha * h[i], wires = i)

            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliX(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliX(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)
        elif (self.gate == 'Z'):
            for i in range(dim): #initial state
                qml.H(wires = i) 
            for layer_index in range (params.shape[0]): 
                if (layer_index % 2 == 0): #mixer layer
                    beta = params[layer_index]
                    for i in range(dim):
                        qml.RX(phi = 2 * beta, wires = i)
                else: # cost layer 
                    alpha = params[layer_index]
                    for i in range (dim):
                        for j in range (i + 1, dim):
                            qml.IsingZZ(phi = -2 * alpha * J[i,j], wires = [i, j]) 
                    for i in range (dim): 
                        qml.RZ(phi = -2 * alpha * h[i], wires = i)

            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliZ(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliZ(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)
    def _MA_problem_inspired_circuit (self, J : np.ndarray, h : np.ndarray, H_cost : np.tensor, params : np.ndarray, start : str = '', mode : str = 'state', initial_angles = []):
        
        '''
        Multi-angle problem-inspired ansatz circuit
        
        **Initial state:** put all qubits in $\ket{+}$ using H gate (for Z basis) or do nothing (for X basis) 

        **Mixer layer:** implement RX/RZ multi-angle parametrized rotations on each qubit 

        **Cost layer:** implement Hamiltonian of XX or ZZ rotations depending on gate parameter and add some X or Z respectively
        
        Parameters 
        ---------
        J : np.ndarray 
            Ising matrix 
        h : np.ndarray
            Ising vector 
        H_cost : np.tensor 
            Cost hamiltonian 
        params : np.ndarray 
            -Cold: array of optimizable parameters with shape([1, depth * (dim + 1)])
            -Warm: array of optimizable parameters with shape([1, depth * (2 * dim + 1)])
        mode : str 
            state or expectation 
        gate : str 
            Z or X depending on entaglment gate ZZ or XX respectively 

        '''
        dim = J.shape[0]
        
        if (self.gate == 'X'): 
            for layer_index in range (params.shape[0]): 
                if (layer_index % (dim + 1) < dim): #mixer layer
                    beta = params[layer_index]
                    qml.RZ(phi = 2 * beta, wires = layer_index % (dim + 1))
                else: # cost layer 
                    alpha = params[layer_index]
                    for i in range (dim):
                        for j in range (i + 1, dim):
                            qml.IsingXX(phi = -2 * alpha * J[i,j], wires = [i, j]) 
                    for i in range (dim): 
                        qml.RX(phi = -2 * alpha * h[i], wires = i)

            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliX(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliX(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)
        elif (self.gate == 'Z'):
            if (start == ''):
                for i in range(dim): #cold initial state
                    qml.H(wires = i) 
            elif (start == 'WS'):
                for i in range(dim): #cold initial state
                    qml.RY(phi = initial_angles[i], wires = i) 

            if (start == ''):
                for layer_index in range (params.shape[0]): #cold layers
                    if (layer_index % (dim + 1) < dim): #cold mixer layer
                        beta = params[layer_index]
                        qml.RX(phi = 2 * beta, wires = layer_index % (dim + 1))
                    else: # cold cost layer 
                        alpha = params[layer_index]
                        for i in range (dim):
                            for j in range (i + 1, dim):
                                qml.IsingZZ(phi = -2 * alpha * J[i,j], wires = [i, j]) 
                        for i in range (dim): 
                            qml.RZ(phi = -2 * alpha * h[i], wires = i)
            elif (start == 'WS'): #warm layers
                for layer_index in range (0, params.shape[0], 2): 
                    if (layer_index % (2 * dim + 1) < 2 * dim): #warm mixer layer
                        alpha = params[layer_index]
                        beta = params[layer_index + 1]
                        qml.RZ(phi = - alpha, wires = layer_index % (2 * dim + 1) // 2)
                        qml.RY(phi = - 2 * beta, wires = layer_index % (2 * dim + 1) // 2)
                        qml.RZ(phi = -2 * alpha, wires = layer_index % (2 * dim + 1) // 2)
                    else: #warm cost layer 
                        alpha = params[layer_index]
                        for i in range (dim):
                            for j in range (i + 1, dim):
                                qml.IsingZZ(phi = -2 * alpha * J[i,j], wires = [i, j]) 
                        for i in range (dim): 
                            qml.RZ(phi = -2 * alpha * h[i], wires = i)
            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliZ(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliZ(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)
                case 'ket':
                    return qml.state()
    
    def _hardware_efficient_circuit  (self, J : np.ndarray, h : np.ndarray, H_cost : np.tensor, params : np.ndarray, mode : str = 'state'):
        ''' 
        hardware-efficient one-angle cost layer parametrization ansatz circuit 
        
        **Initial state:** put all qubits in $\ket{+}$ using H gate (for Z basis) or do nothing (for X basis) 

        **Mixer layer:** implement RY (for Z basis) and RY (for X basis) rotations on each qubit 

        **Cost layer:** cycle-fixed-action entanglement with XX or ZZ 

        Parametes 
        ---------
        J : np.ndarray 
            Ising matrix 
        h : np.ndarray
            Ising vector 
        H_cost : np.tensor 
            Cost hamiltonian 
        params : np.ndarray 
            array of optimizable parameters with shape([2 * depth, 1]) because Mixer + Cost
        mode : str 
            state or expectation 
        gate : str 
            Z or X depending on entaglment gate ZZ or XX respectively 
        '''

        dim = J.shape[0]
        if (self.gate == 'X'):
            # initial state - ket |0> on all qubits 
            
            for layer_index in range(params.shape[0]):
                if (layer_index % 2 == 0): #mixer layer
                    beta = params[layer_index]
                    for i in range(dim): 
                        qml.RY(phi = beta, wires = i)

                else: 
                    alpha = params[layer_index]
                    for i in range (dim):
                        wire1 = i
                        if (wire1 == dim - 1):
                            wire2 = 0
                        else: 
                            wire2 = wire1 + 1
                        qml.IsingXX(phi = alpha, wires = [wire1, wire2])

            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliX(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliX(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)
        
        if (self.gate == 'Z'):
            
            # initial state - ket |+> on all qubits 
            for layer_index in range(params.shape[0]):
                if (layer_index % 2 == 1):
                    beta = params[layer_index]
                    for i in range(dim): 
                        qml.RY(phi = beta, wires = i)

                else: 
                    alpha = params[layer_index]
                    for i in range (dim):
                        wire1 = i
                        if (wire1 == dim - 1):
                            wire2 = 0
                        else: 
                            wire2 = wire1 + 1
                        qml.IsingZZ(phi = alpha, wires = [wire1, wire2])

            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliZ(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliZ(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)
    def _MA_hardware_efficient_circuit (self, J : np.ndarray, h : np.ndarray, H_cost : np.tensor, params : np.ndarray, start : str = '', mode : str = 'state', initial_angles = []):

        '''
        Multi-angle hardware-efficient ansatz circuit 
        
        **Initial state:** put all qubits in $\ket{+}$ using H gate (for Z basis) or do nothing (for X basis) 

        **Mixer layer**: implement RX (for Z basis) and RZ (for X basis) rotations on each qubit with multiangle parametrization

        **Cost layer**: cycle-fixed-action entanglement with XX or ZZ with multi angle parametrization

        Parametes 
        ---------
        J : np.ndarray 
            Ising matrix 
        h : np.ndarray
            Ising vector 
        H_cost : np.tensor 
            Cost hamiltonian 
        params : np.ndarray 
            -Cold: array of optimizable parameters with shape([2 * depth, dim]) because Mixer + Cost
            -Warm: array of optimizable parameters with shape([3 * depth, dim]) because warm-Mixer + Cost
        mode : str 
            state or expectation 
        gate : str 
            Z or X depending on entaglment gate ZZ or XX respectively 
        '''

        dim = J.shape[0]
        if (self.gate == 'X'):
            # initial state - ket |0> on all qubits 
            for layer_index in range(params.shape[0]):
                if (layer_index % 2 == 0): #mixer layer
                    for i in range(dim): 
                        qml.RY(phi = params[layer_index, i], wires = i)

                else: #cost layer
                    for i in range (dim):
                        wire1 = i
                        if (wire1 == dim - 1):
                            wire2 = 0
                        else: 
                            wire2 = wire1 + 1
                        qml.IsingXX(phi = params[layer_index, i], wires = [wire1, wire2])

            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliX(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliX(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)
        
        if (self.gate == 'Z'):

            if (start == ''):
                for i in range(dim): #cold initial state ket |+> on all qubits
                    qml.H(wires = i) 
            elif (start == 'WS'):
                for i in range(dim): #cold initial state
                    qml.RY(phi = initial_angles[i], wires = i) 
            
            for layer_index in range(params.shape[0]):
                
                if (start == 'WS'): # warm
                    if (layer_index % 3 == 0): 
                        for i in range(dim): 
                            alpha = params[layer_index, i]
                            beta = params[layer_index + 1, i]
                            qml.RZ(phi = -alpha, wires = i)
                            qml.RY(phi = -2 * beta, wires = i)
                            qml.RZ(phi = -2 * alpha, wires = i)
                        
                    elif (layer_index % 3 == 2): # cost layer
                        for i in range (dim):
                            wire1 = i
                            if (wire1 == dim - 1):
                                wire2 = 0
                            else: 
                                wire2 = wire1 + 1
                            qml.IsingZZ(phi = params[layer_index, i], wires = [wire1, wire2])
                else: # cold 
                    if (layer_index % 2 == 0): #mixer layer 
                        for i in range(dim): 
                            qml.RY(phi = params[layer_index, i], wires = i)

                    else: #cost layer
                        for i in range (dim):
                            wire1 = i
                            if (wire1 == dim - 1):
                                wire2 = 0
                            else: 
                                wire2 = wire1 + 1
                            qml.IsingZZ(phi = params[layer_index, i], wires = [wire1, wire2])
            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliZ(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliZ(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(-H_cost)
                
    def _MA_alternating_layer_circuit (self, J : np.ndarray, h : np.ndarray, H_cost : np.tensor, params : np.ndarray, mode : str = 'state'): 
        ''' 
        Alternating-layer structured hardware-efficient ansatz circuit 
        
        **Initial state:** put all qubits in $\ket{+}$ using H gate (for Z basis) or do nothing (for X basis) 

        Do following for half of qubits alternating on each layer:

        **Mixer layer:** implement RX (for Z basis) and RZ (for X basis) rotations on each qubit 
       
        **Cost layer:** cycle-fixed-action entanglement with XX or ZZ 

        Parametes 
        ---------
        J : np.ndarray 
            Ising matrix 
        h : np.ndarray
            Ising vector 
        H_cost : np.tensor 
            Cost hamiltonian 
        params : np.ndarray 
            array of optimizable parameters with shape([2 * depth, dim]) because Mixer + Cost
        mode : str 
            state or expectation 
        gate : str 
            Z or X depending on entaglment gate ZZ or XX respectively 
        '''

        dim = J.shape[0]
        sub_dim = dim // 2
        shift = dim // 4
        if (self.gate == 'X'):
            # initial state - ket |0> on all qubits 
            start = 0
            for layer_index in range(params.shape[0]):
                if (layer_index % 2 == 0): #mixer layer
                    for i in range(dim): 
                        qml.RY(phi = params[layer_index, i], wires = i)

                else: # alternating cost layer 
                    start %= dim 
                    for i in range (start, start + sub_dim):
                        wire1 = i % dim 
                        wire2 = (i + 1) % dim
                        qml.IsingXX(phi =  params[layer_index, wire1], wires = [wire1, wire2])

                    for i in range (start + sub_dim, start + dim):
                        wire1 = i % dim
                        wire2 = (i + 1) % dim 
                        qml.IsingXX(phi = params[layer_index, wire1], wires = [wire1, wire2])

                    start += shift

            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliX(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliX(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)
        
        if (self.gate == 'Z'):
            
            # initial state - ket |0> on all qubits 
            start = 0
            for layer_index in range(params.shape[0]):
                if (layer_index % 2 == 0): #mixer layer
                    for i in range(dim): 
                        qml.RY(phi = params[layer_index, i], wires = i)

                else: # alternating cost layer 
                    start %= dim 
                    for i in range (start, start + sub_dim):
                        wire1 = i % dim 
                        wire2 = (i + 1) % dim
                        qml.IsingZZ(phi =  params[layer_index, wire1], wires = [wire1, wire2])

                    for i in range (start + sub_dim, start + dim):
                        wire1 = i % dim
                        wire2 = (i + 1) % dim 
                        qml.IsingZZ(phi = params[layer_index, wire1], wires = [wire1, wire2])

                    start += shift

            match mode : 
                case 'state':
                    state = np.array([qml.expval(qml.PauliZ(i)) for i in range (dim)])
                    return state    
                case 'samples': 
                    state = np.array([qml.sample(qml.PauliZ(i)) for i in range (dim)])
                    return state
                case 'expectation':
                    return qml.expval(H_cost)

    def checkQUBO (self, Q : np.ndarray): 
        '''
        Solve QUBO form using bruteforce
        
        Parameters 
        ----------
        Q : np.ndarray 
            ising matrix 
        Returns
        -------
        min_val : float 
            minimum energy 
        sol_states : list[np.ndarray]
            a list of optimal bitvectors
        end - start : float 
            time the bruteforce took
        '''

        start = time.time()
        min_val = 1e9
        sol_states = []
        dim = Q.shape[0]

        #with tqdm.tqdm(total = 2 * dim, desc="Brute forcing Ising QUBO") as pbar:
        for bits in product([0, 1], repeat = dim):
            bits = np.array(bits)
            val = self._calculateQUBO(Q,bits)
            if (val < min_val): 
                min_val = val
            #pbar.update(1)
        
        for bits in product([0, 1], repeat = dim):
            bits = np.array(bits)
            val = self._calculateQUBO(Q, bits) 
            if (val == min_val): 
                sol_states.append(bits)

            #pbar.update(1)
        
        end = time.time()
        
        return min_val, sol_states, end - start




    def checkIsing (self, J : np.ndarray, h : np.ndarray): 
        '''
        Solve Ising form using bruteforce
        
        Parameters 
        ----------
        J : np.ndarray 
            ising matrix 
        h : np.ndarray 
            ising vector

        Returns
        -------
        min_val : float 
            minimum energy
        sol_states : list[np.ndarray]
            a list of optimal states
        end - start : float 
            time the bruteforce took
        '''

        start = time.time()
        min_val = 1e9
        sol_states = []
        dim = J.shape[0]

        #with tqdm.tqdm(total = 2 * dim, desc="Brute forcing Ising QUBO") as pbar:
        for bits in product([1, -1], repeat = dim):
            bits = np.array(bits)
            val = self._calculateIsing(J,h,bits)
            if (val < min_val): 
                min_val = val
            #pbar.update(1)
        
        for bits in product([1, -1], repeat = dim):
            bits = np.array(bits)
            val = self._calculateIsing(J,h, bits) 
            if (val == min_val): 
                sol_states.append(bits)

            #pbar.update(1)
        
        end = time.time()
        
        return min_val, sol_states, end - start


    def checkMaxcut (self, W: np.ndarray):
        
        '''
        Solve MaxCut problem using bruteforce
        
        Parameters 
        ----------
        W : np.ndarray 
            MaxCut graph
        Returns
        -------
            max_val : float 
                maximum-cut value 
            sol_states : list[np.ndarray]
                a list of optimal cuts
            end - start : float 
                time the bruteforce took
        '''

        start = time.time()

        max_val = -1e9
        sol_states = []
        dim = W.shape[0] - 1

        for bits in product([1, -1], repeat = dim):
            bits = np.array(bits)
            val = self._calculateMaxcut(W,bits)
            if (val > max_val): 
                max_val = val
            #pbar.update(1)
        
        for bits in product([1, -1], repeat = dim):
            bits = np.array(bits)
            val = self._calculateMaxcut(W,bits) 
            if (val == max_val): 
                sol_states.append(bits)

        end = time.time()

        return max_val, sol_states, end - start 

    def solveGWMaxcut(self, W : np.ndarray): 

        ''' 
        Solve MaxCut problem by implementing Goeman-Williamson relaxation 
        It's wlg assumed that zero vertice has s_0 = +1

        Parameters
        ----------
        W : np.ndarray 
            graph weight matrix 
        round : bool 
            does cutvector need rounding
        Returns
        -------
        cutvector: np.ndarray
            a cutstring coresponding to the optimal cut
        '''

        start = time.time()
        #Symmetrizing matrix 
        W = (W + W.T)
        try:
            W = W.numpy()
        except:
            pass
        n = W.shape[0]
        
        np.fill_diagonal(W, 0)  

        # Defining SDP problem 
        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0] 
        constraints += [X[i, i] == 1 for i in range(n)] 

        # objective: min trace(W @ X)
        objective = cp.Minimize(cp.trace(W @ X))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        S = X.value

        normal = np.random.uniform(low = -1, high = 1, size = (1, W.shape[1]))[0]

        # Scalar products vector equals normal.T @ S 
        #print(normal.T @ S)
        end = time.time()
        # if (round): cutvector = np.where (normal.T @ S > 0, 1, -1)


        cutvector = (normal.T @ S) / np.linalg.norm(normal.T @ S)

        # cutvalue = self._calculateMaxcut(W, cutvector, needs0 = False)
        #print(cutvector)
        return cutvector [1:]
    
    def _warmStart(self, Q: np.ndarray, eps: float = 0.25):
        
        '''
        Prepare initital angles vector for WS-QAOA 

        Parameters
        ----------
        Q : np.ndarray 
            QUBO matrix 
        eps : float = 0.25
            a regularization parameter
        Returns
        -------
        intiital_angles : np.ndarray 
        '''

        W, C1 = self._maxcutForm(Q)

        cutvector = self.solveGWMaxcut(W)

        initital_angles = np.zeros(Q.shape[0])
        for i in range(len(cutvector)):
            if (cutvector[i] <= 1 - eps) and (eps <= cutvector[i]):
                initital_angles[i] = 2 * np.arcsin(np.sqrt(cutvector[i]))
            elif (cutvector[i] <= eps):  
                initital_angles[i] = 2 * np.arcsin(np.sqrt(eps))
            else: 
                initital_angles[i] = 2 * np.arcsin(np.sqrt(1 - eps))

        return initital_angles

    def _maxcutForm (self, Q: np.ndarray): 
        '''
        convert QUBO problem to an equivalent maxcut problem 

        Parameters
        ----------
        Q : np.ndarray
            QUBO matrix with generally non-zero diagonal elements

        Returns 
        -------
        w : np.ndarray 
            MaxCut problem +-1-marked vertices externed for s_0 = +1 weight matrix
        C1 : np.ndarray
            constant caused by transformation from QUBO 
        '''

        c = np.array([0.5 * Q[i,i] for i in range (Q.shape[0])])
        
        for i in range (Q.shape[0]): Q[i,i] = 0  

        Qsum = 0

        for j in range(Q.shape[0]):
            for i in range (i + 1, Q.shape[1]): 
                Qsum += Q[i,j]

        C1 = 1/4 * Qsum + 1/2 * c.sum()

        w = np.zeros(shape = (Q.shape[0] + 1, Q.shape[1] + 1))
        
        for i in range(1, Q.shape[0] + 1): 
            for j in range (i + 1, Q.shape[1] + 1):
                w[i, j] = 1/4 * Q[i - 1, j - 1]

        for j in range (1, Q.shape[1] + 1):
            sqij = 0 
            sqji = 0 

            for i in range (0, j - 1): 
                sqij += Q[i, j - 1]
            for i in range (j, Q.shape[0]):
                sqji += Q[j - 1, i] 

            w[0, j] = 1/4 * (sqij + sqji) + 0.5 * c[j - 1]
    
        return w, C1

    def _isingForm (self, Q: np.ndarray) -> list[np.ndarray, np.ndarray]:
        '''Prepare Ising form of QUBO problem
        
        Parameters
        ----------
        Q: np.ndarray
            matrix Q of original QUBO problem

        Returns
        -------
        Ising matrix J and vector h in 
        '''
        J = np.zeros(Q.shape)
        h = np.zeros(Q.shape[0])
        for i in range (Q.shape[0]):
            q_sum = 0
            for j in range (Q.shape[1]): 
                J[i][j] = -(Q[i][j] * 1/4) * (1 - (i == j)) 
                q_sum += Q[i][j]
                
            h[i] = -0.25 * q_sum 
            
        return J, h 
    
    def isingHamiltonian (self, J : np.ndarray, h : np.ndarray) -> np.tensor:

        """
        Prepare XX gates based cost Hamiltonian from the given matrix
        Parameters
        ----------
        J : np.ndarray
            Ising matrix J, shape([dim,dim])
        h : np.ndarray
            Ising vector h, shape([1, dim])
        ----------
        Returns: 
        Ising Hamiltonian based on specialized gates
        """ 

        H = 0 * qml.I(0)
        dim = J.shape[0]
        if (self.gate == 'X'):
            for i in range (dim):
                for j in range (dim):
                    if (i != j):
                        XXij = qml.X(i) @ qml.X(j)
                        for k in range (dim):
                            if (k != i) and (k != j):
                                XXij @= qml.I(k)
                
                        H += -0.5 * J[i][j] * XXij 
                X_h = -h[i] * qml.X(i) 
                for k in range(dim): 
                    if (k != i):
                        X_h @= qml.I(k)
                H += X_h
                            
            return H    
        elif (self.gate == 'Z'):
            for i in range (dim):
                for j in range (dim):
                    if (i != j):
                        ZZij = qml.Z(i) @ qml.Z(j)
                        for k in range (dim):
                            if (k != i) and (k != j):
                                ZZij @= qml.I(k)
                
                        H += -0.5 * J[i][j] * ZZij 
                Z_h = -h[i] * qml.Z(i) 
                for k in range(dim): 
                    if (k != i):
                        Z_h @= qml.I(k)
                H += Z_h
                            
            return H

    
        

    def solve (self, Q: np.ndarray, depth : int, sol : float, stepsize: float, log = False): 
        
        '''
            Circuit optimizer

            Parameters
            ----------
            Q : np.ndarray
                QUBO matrix 
            depth : int 
                how many (cost + mixer) layers would be implemented
            sol : float
                truth solution for the optimization problem 
            optimizer_hyperparams : list[float]
                a list of hyperparameters for ADAM optimizer to use 
            logs : bool 
                should i print logs 
            Returns 
            -------
            minsol : float 
                Ising problem min value
            bitstring : np.ndarray 
                Ising problem solution bitstring
            minstate : np.ndarray
                Quantum solution state
            quantum_iterations : int 
                a number of quantum iterations needed to converge or reach the iterations limit
            itererr : list[len(accepterr)]
                a list of iterations counts to reach listed in accepterr errors
            end - start : float 
                time it took 
            baren_flag  : bool 
                have stopped because of a baren plateau
            baren_index : int 
                iteration number on which baren plateau occured
        '''

        # ToDo
        # [ ] Add Baren-plateu criteria - if energies are allclose 
        # [ ] Add best approximation 
            # infinite shots device or bitstring to ket

        #Converting to Ising problem
        J, h = self._isingForm(Q)
        H_cost = self.isingHamiltonian(J, h)
        dim = J.shape[0]

        #setting the optimizer
        optimizer = qml.AdamOptimizer(stepsize)

        

        #prepating circuits

        initial_angles = np.zeros(Q.shape[0])
        if (self.preprocessing == 'WS'): 
            initial_angles = self._warmStart(Q) #prepare warm initial angles if needed 
            if (log): print(f'initial_angles: {initial_angles}')
        low, high = -1,1
        match self.ansatz: 
            case 'CIA':
                if (self.preprocessing == 'WS'):
                    params = np.random.uniform(size = (depth, 2 * dim), low = low, high = high, requires_grad = True)
                else:
                    params = np.random.uniform(size = (depth, dim), low = low, high = high, requires_grad = True) 
                circuit = qml.QNode(self._classical_inspired_circuit, self.dev)
            case 'PIA': 
                params = np.random.uniform(size = (1, 2 * depth), low = low, high = high, requires_grad = True)[0]
                circuit = qml.QNode(self._problem_inspired_circuit, self.dev)
            case 'MA-PIA': 
                if (self.preprocessing == 'WS'):
                    params = np.random.uniform(size = (1, (2 * dim + 1) * depth), low = low, high = high, requires_grad = True)[0]
                else:
                    params = np.random.uniform(size = (1, (dim + 1) * depth), low = low, high = high, requires_grad = True)[0]
                circuit = qml.QNode(self._MA_problem_inspired_circuit, self.dev) 
            case 'HEA': 
                params = np.random.uniform(size = (1, 2 * depth), low = low, high = high, requires_grad = True)[0]
                circuit = qml.QNode(self._hardware_efficient_circuit, self.dev)
            case 'MA-HEA':     
                if (self.preprocessing == 'WS'):
                    params = np.random.uniform(size = (3 * depth, dim), low = low, high = high, requires_grad = True)
                else:
                    params = np.random.uniform(size = (2 * depth, dim), low = low, high = high, requires_grad = True)
                circuit = qml.QNode(self._MA_hardware_efficient_circuit, self.dev)
            case 'MA-ALT':
                params = np.random.uniform(size = (2 * depth, dim), low = low, high = high, requires_grad = True)
                circuit = qml.QNode(self._MA_alternating_layer_circuit, self.dev)
                
        def cost_circuit(params):
            if (self.postprocessing == 'CVaR'):
                cost = self._CVaR_expectation(J,h,samples = circuit(J = J, h = h, H_cost = H_cost, params = params, start = self.preprocessing, mode = 'samples', initial_angles = initial_angles))
                #print (cost)
                return cost
            else: 
                return circuit(J = J, h = h, H_cost = H_cost, params = params, start = self.preprocessing, mode = 'expectation', initial_angles = initial_angles)
        def energy_circuit(params): 
            return circuit(J = J, h = h, H_cost = H_cost, params = params, start = self.preprocessing, mode = 'expectation', initial_angles = initial_angles)
        def state_circuit(params):
            return circuit (J = J, h = h, H_cost = H_cost, params = params, start = self.preprocessing, mode = 'state', initial_angles = initial_angles)
        def ket_circuit(params): 
            return circuit (J = J, h = h, H_cost = H_cost, params = params, start = self.preprocessing, mode = 'ket', initial_angles = initial_angles)
        
        def cursol_circuit(params): 
            if (self.postprocessing == 'CVaR'):
                cost = self._CVaR_expectation(J,h,samples = circuit(J = J, h = h, H_cost = H_cost, params = params, start = self.preprocessing, mode = 'samples', initial_angles = initial_angles))
                return cost
            else: 
                state = state_circuit(params)
                bitstring = np.where(state > 0, 1, -1)
                return self._calculateIsing(J, h, bitstring)
            
        quantum_iterations = 0
        
        cursol = 1e8
        
        exiterr = self.accepterr[0] # error with which we conclude algortihm to be best-fit
        itererr = [-1] * len(self.accepterr) # list of iterations for fitting in each error 
        
        minsol = 1e8
        minstate = np.zeros(dim)
        minket = np.zeros(2 ** dim)

        start = time.time()

        bitstring = np.array([])

        energy_gradients = queue.Queue(maxsize = self.baren_threshold)
        energy_mean_grad = 1e8
        baren_flag = False 
        preenergy = 0
        preparams = []
        while (quantum_iterations <= self.iteration_limit) and (not np.allclose(cursol, sol)): #np.abs((cursol - sol) / sol) > exiterr): 
            #print(params)
        
            params = optimizer.step(cost_circuit, params)
            state = state_circuit(params)
            curenergy = energy_circuit(params)
            cursol = cursol_circuit(params)
            
            
            if (cursol < minsol):
                minsol = cursol 
                minstate = state
                bitstring = np.where(minstate > 0, 1, -1)
            
            #calculate energy-gradient
            
            if (self.baren_check) and (quantum_iterations > 0):
                energy_grad = np.abs(curenergy - preenergy) #/ np.linalg.norm(preparams - params)
                energy_gradients.put(energy_grad)
                energy_mean_grad = np.mean(list(energy_gradients.queue))

                if (energy_gradients.qsize() == self.baren_threshold) and (np.abs(energy_mean_grad / curenergy) < self.baren_rerr / (self.iterations_limit - quantum_iterations)):
                    baren_flag = True 
                    #print('baren detected')
                    break

            if (quantum_iterations > 1) and (energy_gradients.qsize() == self.baren_threshold): 
                energy_gradients.get()

            quantum_iterations += 1 

            if (log) and (quantum_iterations % 1 == 0): 
                print (f'Ansatz: {self.ansatz} Iteration: {quantum_iterations} | State: {state} | Sol: {sol} | Cursol: {cursol} | Curenergy: {curenergy} | Preenergy: {preenergy} | Baren_queue: {list(energy_gradients.queue)} | {energy_mean_grad}')
            
            #calculating error
            err = np.abs((cursol - sol) / sol)

            for ierr in range(len(itererr)):
                if (self.accepterr[ierr] >= err) and (itererr[ierr] == -1):
                    itererr[ierr] = quantum_iterations 
           
            
            preenergy = curenergy
            preparams = params

        #print(f'Converged with vector {bitstring} and min energy {cursol}')
        end = time.time()
        baren_index = quantum_iterations - self.baren_threshold
        return minsol, bitstring, minstate, quantum_iterations, itererr, end - start, baren_flag, baren_index
        

   
