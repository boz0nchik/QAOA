import random 

import pennylane as qml
from pennylane import numpy as np 
# import numpy as np

class Preparator: 
    def __init__ (self): 
        '''
        Class for preparing QUBO matrices with a certain set of parameters
        '''
    def rankPreparator (self, size : int, rank : int, value_range : list[2] = [-1, 1], cycles : int = 3): 
        ''' 
        Preapare matrix size x size with a certain even rank

        Parameters
        ----------
        size : int
        rank : int 
        value_range : list [2] = [0,1]
            [min, max] values
        
        Returns
        -------
        matrix : np.ndarray
            generated matrix size x size
        '''

        cur_rank = 0
        rank //= 2

        while cur_rank != rank: # almost always done in 1 iteration
            matrix = matrix = np.random.uniform(low = value_range[0], high = value_range[1],size = (rank, rank))
            cur_rank = np.linalg.matrix_rank(matrix)

        for i in range (rank + 1, size + 1): 
            matrix = np.vstack([matrix, np.zeros(i - 1)])
            matrix = np.hstack([matrix, (np.zeros(i)).reshape(i,1)])

        #implementing I type: [1,i,j,k]  matrix[i] += matrix[j] * k
    
        trans_len = size * 10

        for cyc in range(cycles):
            
            for _ in range (trans_len):
                trans_type = random.randint(1,2)

                if (trans_type == 1):
                
                    i = random.randint(0,size-1)
                    j = random.randint(0,size-1)
                    k = random.uniform(-1,1)
                    
                    matrix[i] += matrix[j] * k

            matrix = matrix.T
            for _ in range (trans_len):
                trans_type = random.randint(1,2)

                if (trans_type == 1):
                
                    i = random.randint(0,size-1)
                    j = random.randint(0,size-1)
                    k = random.uniform(0.01,1)
                    
                    matrix[i] += matrix[j] * k

            matrix = matrix.T
        # Normalize 

        norm = max(value_range, key = np.abs) / max(matrix.reshape(-1,), key = np.abs)

        matrix *= norm
        
        return (matrix + matrix.T) / 2
        
    def densePreparator (self, size : int, density : int, value_range : list[2] = [-1, 1], eps : float = 0.01): 
        ''' 
        Preapare matrix size x size with a certain dense. 

        Parameters
        ----------
        size : int
        density : int 
        value_range : list [2] = [0,1]
            [min, max] values
        eps : float = 0.05
            0-close values would be eps % from max value and 1-close values would be (1 - eps) % from max value 
        Returns
        -------
        matrix : np.ndarray
            generated matrix size x size
        '''

        ones_count = int(size * size * density)

        matrix = np.random.uniform(low = 0, high = value_range[1] * eps, size = (size, size))

        ones_indices = random.sample(range(size * size), ones_count)

        
        for idx in ones_indices:
            row = idx // size
            col = idx % size
            matrix[row][col] = random.uniform((1 - eps) * value_range[1], value_range[1])
            

        for i in range (size):
            for j in range (size):

                val = random.uniform(value_range[0], value_range[1])
                matrix[i][j] *= val
        
        return (matrix + matrix.T) / 2
    
    def stiffnessPreparator (self, size : int, stiffness : float, value_range : list[2] = [-1, 1]):

        '''
        We use eigenratio absolute value as a measure of QUBO matrix stiffness

        $$\kappa = \frac{|\labmda_max|}{\lambda_min}$$

        '''
        
        real_arr = np.random.uniform(low = value_range[0], high = value_range[1], size = (1, size))[0]

        # imaginary_arr = np.random.uniform(low = value_range[0], high = value_range[1], size = (1, size // 2))[0]

        # complex_arr = real_arr + 1j * imaginary_arr

        min_index = np.argmin(np.abs(real_arr))
        max_index = np.argmax(np.abs(real_arr))
        
        correction = stiffness / (np.abs(real_arr[max_index]) / np.abs(real_arr[min_index]))

        # complex_arr[max_index] *= correction 
        real_arr[max_index] *= correction
        matrix = np.diag(real_arr)
        # imaginary_arr[max_index] *= correction
        # matrix = np.zeros(shape = [size, size])
        
        #prepating a matrix of blocks 
        # (a b)
        # (-b a)


        # for i in range(0,size - size % 2, 2):
        #     matrix[i, i] = real_arr[i // 2] 
        #     matrix[i + 1, i + 1] = real_arr[i // 2]
        #     matrix[i, i + 1] = imaginary_arr[i // 2]
        #     matrix[i + 1, i] = -imaginary_arr[i // 2]  

        # if (size % 2 == 1):
        #     matrix[size - 1, size - 1] = np.abs(complex_arr[min_index])
    
        Temp = np.random.uniform (low = -1, high = 1, size = (size, size))
        
        #print (f'C : {C}')

        C, R = np.linalg.qr(Temp)
        # C = C + C.T 
        # print (f'is ortogonal {np.allclose(C.T, np.linalg.inv(C))}')

        matrix = np.linalg.inv(C) @ matrix @ C
        
        # #Normalize 

        # A = A^T

        # C^T = C^-1 

        # (C^-1 A C) ^ T = C^T A^T (C^-1)^T = C^-1 A C 

        norm = max(value_range, key = np.abs) / max(matrix.reshape(-1,), key = np.abs)

        matrix *= norm

        
        return matrix




    def isingForm (self, Q: np.ndarray) -> list[np.ndarray, np.ndarray]:
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

    def maxcutForm (self, Q: np.ndarray): 
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


    def isingHamiltonian (self, J : np.ndarray, h : np.ndarray, gate : str = 'X') -> np.tensor:

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
        if (gate == 'X'):
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
        elif (gate == 'Z'):
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

    def constrained_data(self, Q: np.ndarray, K: float, M: float) -> np.ndarray:
        """
        Calculates Q coefficient for QUBO model with budget equality constraint.

        Parameters
        ----------
        Q : np.ndarray
            Unconstrained MPT Q matrix, shape([dim,dim]).
        K : float
            Budget size.
        M : float
            Enforced budget penalty coefficient.

        Returns
        -------
        Q : np.ndarray
            Constrained MPT Q matrix, shape([dim,dim]).
        """
        N = Q.shape[0]
        E = np.ones([N, N])
        Id = np.diag(np.ones(N))
        G = E - 2 * K * Id
        Q = Q + M * (G)

        return Q

    def penalty_weight(self, Q: np.ndarray, K: float, type: str | str = 'MOC') -> float:
        """
        Calculates penalty weights from https://arxiv.org/abs/2206.11040 rn for budger equality constraint
    
        Parameters
        ----------
        Q : np.ndarray
            Unconstrained MPT Q matrix, shape([dim,dim]).
        K : int
            Budget size. NOte that K==N/2 doesn't work for some reason
        type : str, optional
            Corresponds to the type of penalty weight from article. Current types:
            MOC: Maximum change in Objective function (most optimal)
            MOMC: Maximum change in Objective function divided by Minimum Constraint function
            VLM: https://www.researchgate.net/publication/342539451_Penalty_and_partitioning_techniques_to_improve_performance_of_QUBO_solvers
            MQC: The Maximum QUBO Coe
    cient
            UB: Upper Bound of the objective function
            
        Returns
        -------
        M : float
            Calculated penalty coefficient.
        """
        N = Q.shape[0]
        E = np.ones([N, N])
        #print(f'E : {E}')
        Id = np.diag(np.ones(N))
        
        if type == 'UB':
            return np.sum(Q)
        elif type == 'MQC':
            return np.max(Q)
            
        G = E -  2 * K * Id  #your penalty form. Can be updated for different penalty types
        #print(f'G : {G}')
        WC = np.zeros([N])
        WG = np.zeros([N])
        Qd = np.diag(np.diag(Q))
        #print(f'Qd : {Qd}')
        Q = Q - Qd
        Gd = np.diag(np.diag(G))
        G = G - Gd
        Qp = np.where(Q >= 0, Q, 0) + Qd
        Qn = np.where(Q <= 0, Q, 0) + Qd
        Gp = np.where(G >= 0, G, 0) + Gd
        Gn = np.where(G <= 0, G, 0) + Gd
        #print(f'Qd : {np.diag(Qd)}, np.sum(Qp) : {np.sum(Qp, axis = 1)}') 
        WC = np.array([np.diag(Qd) + np.sum(Qp, axis = 1), -np.diag(Qd) - np.sum(Qn, axis = 1)]).reshape(1, 2 * N)[0]
        WG = np.array([np.diag(Gd) + np.sum(Gp, axis = 1), -np.diag(Gd) - np.sum(Gn, axis = 1)]).reshape(1, 2 * N)[0]
        #print(f'shape : {WC.shape}')
        #WC = np.max(np.array([-np.diag(Qd) + np.sum(Qp, axis=1), np.diag(Qd) - np.sum(Qn, axis=1)]), axis=0)
        #WG = np.min(np.array([-np.diag(Gd) + np.sum(Gp, axis=1), np.diag(Gd) - np.sum(Gn, axis=1)]), axis=0)
        VLM = np.max(np.abs(WC))
        #print(f'WG : {WG}')
        # print(f'WG > 0 : {WG > 0}')
        # print(f'MC : {WC}')
        # WGp, WCp = [], []
        # for i in range(len(WG)):
        #     if (WG[i] > 0):
        #         WGp.append(WG[i])
        #         WCp.append(WC[i]) 
        WGp = WG[WG > 0]
        WCp = WC[WG > 0]
        WGp = np.array(WGp)
        WCp = np.array(WCp)
        gamma = np.min(WGp)
        #print(gamma)
        MOMC = np.max([1, VLM / gamma])
        MOC = np.max([1, np.max(np.abs(WCp/WGp))])
        
        if type == 'VLM':
            M = VLM
        elif type == 'MOMC':
            M = MOMC
        elif type == 'MOC':
            M = MOC

        return M
