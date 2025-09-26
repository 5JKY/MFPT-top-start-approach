# Revolution Version with Markov Chain Extended: P_B->ReAb, P_A->AbRe, P->AbAb, P'_B->InReAb, P'_A->AbReIn
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve, eigs

def get_steady_state(ria_trans, hx):
    """
    ria_trans: an instance of a transfer matrix class defined in transfer_matrix_reptile.py
    """
    idx_fixed_vect = np.where(np.round(ria_trans.eig6_w.real, decimals=10) == 1)[0][0]
    steady_state = ria_trans.eig6_v[:,idx_fixed_vect].real/(hx*np.sum(ria_trans.eig6_v[:,idx_fixed_vect].real))
    return steady_state

def metro_accept(beta_U_arr):
    Aab = np.ones(beta_U_arr.size-1)
    Aba = np.ones(beta_U_arr.size-1)
    Aac = np.ones(beta_U_arr.size-1)
    Aca = np.ones(beta_U_arr.size-1)

    right_difference = beta_U_arr[1: ]-beta_U_arr[ :-1]
    # if U_i+1 > Ui: A_i,i+1=exp(-diff), A_i+1,i =1
    plus_mask = right_difference > 0
    Aab[plus_mask == True] = np.exp(-right_difference[plus_mask == True])
    # else: A_i,i+1=1, A_i+1,i =exp(diff)
    Aba[plus_mask == False] = np.exp(right_difference[plus_mask == False])

    # left_difference = beta_U_arr[ :-1]-beta_U_arr[1: ]
    left_difference = -right_difference
    # if U_i-1 < Ui: A_i,i-1 =1, A_i-1,i =exp(diff)
    minus_mask = left_difference < 0
    Aca[minus_mask == True] = np.exp(left_difference[minus_mask == True])
    # else: A_i,i-1 =exp(-diff), A_i-1,i =1
    Aac[minus_mask == False] = np.exp(-left_difference[minus_mask == False])

    return Aab, Aba, Aac, Aca

def criteria_accept(beta_U_arr):
    right_difference = beta_U_arr[1: ]-beta_U_arr[ :-1]
    # left_difference = beta_U_arr[: -1]-beta_U_arr[1: ]
    left_difference = -right_difference
    Aab = 1.0/(np.exp(right_difference)+1.0)
    Aba = 1.0/(np.exp(-right_difference)+1.0)
    Aac = 1.0/(np.exp(left_difference)+1.0)
    Aca = 1.0/(np.exp(-left_difference)+1.0)
    return Aab, Aba, Aac, Aca


class TransferMatrix_ReRe:
    def __init__(self, hx, x_arr, beta_U, criteria=0):
        """
        criteria == 0:  Original Metropolis Acceptance Criterion
        criteria == 1:  Analytical Acceptance Criterion

        Transfer matrix won't use time step size ht
        """
        self.hx = hx
        self.x_arr = x_arr
        # beta_U is a function, can be called by beta_U(arr)
        self.beta_U = beta_U
        self.criteria = criteria
        self.trans_mat = self.assemble_matrix()
        self.eig6_w, self.eig6_v = eigs(self.trans_mat)
        idx_fixed_vect = np.where(np.round(self.eig6_w.real, decimals=10) == 1)[0][0]
        self.steady_state = self.eig6_v[:,idx_fixed_vect].real/(hx*np.sum(self.eig6_v[:,idx_fixed_vect].real))
        self.relax_timescale = -1.0/np.log(self.eig6_w[1])

    def assemble_matrix(self):
        beta_U_arr = self.beta_U(self.x_arr)
        if self.criteria == 0: # Typical Metropolis Acceptance Probability
            Aab, Aba, Aac, Aca = metro_accept(beta_U_arr)
            # Assemble transfer matrix (ReRe)
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            return trans_mat
        
        elif self.criteria == 1: # Analytical Acceptance Probability
            Aab, Aba, Aac, Aca = criteria_accept(beta_U_arr)
            # Assemble transfer matrix (ReRe)
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            return trans_mat
        

class TransferMatrix_ReAb:
    def __init__(self, hx, x_arr, beta_U, criteria=0):
        """
        criteria == 0:  Original Metropolis Acceptance Criterion
        criteria == 1:  Analytical Acceptance Criterion

        Transfer matrix won't use time step size ht
        """
        self.hx = hx
        self.x_arr = x_arr
        # beta_U is a function, can be called by beta_U(arr)
        self.beta_U = beta_U
        self.criteria = criteria
        self.trans_mat = self.assemble_matrix()
        self.eig6_w, self.eig6_v = eigs(self.trans_mat)
        idx_fixed_vect = np.where(np.round(self.eig6_w.real, decimals=10) == 1)[0][0]
        self.steady_state = self.eig6_v[:,idx_fixed_vect].real/(hx*np.sum(self.eig6_v[:,idx_fixed_vect].real))
        self.relax_timescale = -1.0/np.log(self.eig6_w[1])

    def assemble_matrix(self):
        beta_U_arr = self.beta_U(self.x_arr)
        if self.criteria == 0: # Typical Metropolis Acceptance Probability
            Aab, Aba, Aac, Aca = metro_accept(beta_U_arr)
            # Assemble transfer matrix (ReAb)
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, -1] = 0
            trans_mat[-1, -1] = 1
            return trans_mat
        
        elif self.criteria == 1: # Analytical Acceptance Probability
            Aab, Aba, Aac, Aca = criteria_accept(beta_U_arr)
            # Assemble transfer matrix (ReAb)
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, -1] = 0
            trans_mat[-1, -1] = 1
            return trans_mat


class TransferMatrix_AbRe:
    def __init__(self, hx, x_arr, beta_U, criteria=0):
        """
        criteria == 0:  Original Metropolis Acceptance Criterion
        criteria == 1:  Analytical Acceptance Criterion

        Transfer matrix won't use time step size ht
        """
        self.hx = hx
        self.x_arr = x_arr
        # beta_U is a function, can be called by beta_U(arr)
        self.beta_U = beta_U
        self.criteria = criteria
        self.trans_mat = self.assemble_matrix()
        self.eig6_w, self.eig6_v = eigs(self.trans_mat)
        idx_fixed_vect = np.where(np.round(self.eig6_w.real, decimals=10) == 1)[0][0]
        self.steady_state = self.eig6_v[:,idx_fixed_vect].real/(hx*np.sum(self.eig6_v[:,idx_fixed_vect].real))
        self.relax_timescale = -1.0/np.log(self.eig6_w[1])

    def assemble_matrix(self):
        beta_U_arr = self.beta_U(self.x_arr)
        if self.criteria == 0: # Typical Metropolis Acceptance Probability
            Aab, Aba, Aac, Aca = metro_accept(beta_U_arr)
            # Assemble transfer matrix (AbRe)
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, 0] = 0
            trans_mat[0, 0] = 1
            return trans_mat
        
        elif self.criteria == 1: # Analytical Acceptance Probability
            Aab, Aba, Aac, Aca = criteria_accept(beta_U_arr)
            # Assemble transfer matrix (AbRe)
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, 0] = 0
            trans_mat[0, 0] = 1
            return trans_mat


class TransferMatrix_AbAb:
    def __init__(self, hx, x_arr, beta_U, criteria=0):
        """
        criteria == 0:  Original Metropolis Acceptance Criterion
        criteria == 1:  Analytical Acceptance Criterion

        Transfer matrix won't use time step size ht
        """
        self.hx = hx
        self.x_arr = x_arr
        # beta_U is a function, can be called by beta_U(arr)
        self.beta_U = beta_U
        self.criteria = criteria
        self.trans_mat = self.assemble_matrix()
        self.eig6_w, self.eig6_v = eigs(self.trans_mat)
        idx_fixed_vect = np.where(np.round(self.eig6_w.real, decimals=10) == 1)[0][0]
        self.steady_state = self.eig6_v[:,idx_fixed_vect].real/(hx*np.sum(self.eig6_v[:,idx_fixed_vect].real))
        self.relax_timescale = -1.0/np.log(self.eig6_w[1])

    def assemble_matrix(self):
        beta_U_arr = self.beta_U(self.x_arr)
        if self.criteria == 0: # Typical Metropolis Acceptance Probability
            Aab, Aba, Aac, Aca = metro_accept(beta_U_arr)
            # Assemble transfer matrix (AbAb)
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, 0] = 0
            trans_mat[0, 0] = 1
            trans_mat[:, -1] = 0
            trans_mat[-1, -1] = 1
            return trans_mat
        
        elif self.criteria == 1: # Analytical Acceptance Probability
            Aab, Aba, Aac, Aca = criteria_accept(beta_U_arr)
            # Assemble transfer matrix (AbAb)
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, 0] = 0
            trans_mat[0, 0] = 1
            trans_mat[:, -1] = 0
            trans_mat[-1, -1] = 1
            return trans_mat
    

class TransferMatrix_InReAb:
    def __init__(self, hx, x_arr, beta_U, criteria=0):
        """
        criteria == 0:  Original Metropolis Acceptance Criterion
        criteria == 1:  Analytical Acceptance Criterion

        Transfer matrix won't use time step size ht
        """
        self.hx = hx
        self.x_arr = x_arr
        # beta_U is a function, can be called by beta_U(arr)
        self.beta_U = beta_U
        self.criteria = criteria
        self.trans_mat = self.assemble_matrix()
        self.eig6_w, self.eig6_v = eigs(self.trans_mat)
        idx_fixed_vect = np.where(np.round(self.eig6_w.real, decimals=10) == 1)[0][0]
        self.steady_state = self.eig6_v[:,idx_fixed_vect].real/(hx*np.sum(self.eig6_v[:,idx_fixed_vect].real))
        self.relax_timescale = -1.0/np.log(self.eig6_w[1])

    def assemble_matrix(self):
        beta_U_arr = self.beta_U(self.x_arr)
        if self.criteria == 0: # Typical Metropolis Acceptance Probability
            Aab, Aba, Aac, Aca = metro_accept(beta_U_arr)
            # Assemble transfer matrix (InReAb), recycle to left boundary
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, -1] = 0
            trans_mat[0, -1] = 1
            return trans_mat
        
        elif self.criteria == 1: # Analytical Acceptance Probability
            Aab, Aba, Aac, Aca = criteria_accept(beta_U_arr)
            # Assemble transfer matrix (InReAb), recycle to left boundary
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, -1] = 0
            trans_mat[0, -1] = 1
            return trans_mat


class TransferMatrix_AbReIn:
    def __init__(self, hx, x_arr, beta_U, criteria=0):
        """
        criteria == 0:  Original Metropolis Acceptance Criterion
        criteria == 1:  Analytical Acceptance Criterion

        Transfer matrix won't use time step size ht
        """
        self.hx = hx
        self.x_arr = x_arr
        # beta_U is a function, can be called by beta_U(arr)
        self.beta_U = beta_U
        self.criteria = criteria
        self.trans_mat = self.assemble_matrix()
        self.eig6_w, self.eig6_v = eigs(self.trans_mat)
        idx_fixed_vect = np.where(np.round(self.eig6_w.real, decimals=10) == 1)[0][0]
        self.steady_state = self.eig6_v[:,idx_fixed_vect].real/(hx*np.sum(self.eig6_v[:,idx_fixed_vect].real))
        self.relax_timescale = -1.0/np.log(self.eig6_w[1])
        
    def assemble_matrix(self):
        beta_U_arr = self.beta_U(self.x_arr)
        if self.criteria == 0: # Typical Metropolis Acceptance Probability
            Aab, Aba, Aac, Aca = metro_accept(beta_U_arr)
            # Assemble transfer matrix (AbReIn), recylce to right boundary
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, 0] = 0
            trans_mat[-1, 0] = 1
            return trans_mat
        
        elif self.criteria == 1: # Analytical Acceptance Probability
            Aab, Aba, Aac, Aca = criteria_accept(beta_U_arr)
            # Assemble transfer matrix (AbReIn), recylce to right boundary
            A0_ghost, Am_ghost = 0, 0
            ghost_Aac = np.append(A0_ghost, Aac)
            ghost_Aab = np.append(Aab, Am_ghost)
            main_diag = 2.0*np.ones(self.x_arr.size)-ghost_Aac-ghost_Aab

            trans_mat = 0.5*(
                np.diag(Aca, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(Aba, k=1)
                )
            trans_mat[:, 0] = 0
            trans_mat[-1, 0] = 1
            return trans_mat

    