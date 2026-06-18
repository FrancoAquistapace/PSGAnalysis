# Import modules
import pandas as pd
import numpy as np



# ------ I/O FUNCTIONS -------





# ------ FEATURE FUNCTIONS ------
# Function to calculate difference in amino acid composition
def group_composition(sequence_list, amino_group, normalize=False):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure total count.
        normalize : bool (optional)
            Whether to normalize the results by the length
            of each sequence or not. Default is False.
    Output:
        Returns a list of values, indicating the amount
        of amino acids (or the fraction if normalize=True)
        from the given group in each input sequence.
    '''
    # Init results
    res = []

    # Operate for each sequence
    for s in sequence_list:
        s_arr = pd.Series(list(s))
        
        count = np.sum(s_arr.isin(amino_group))
        if normalize:
            count = count / len(s)

        res.append(count)

    return res


# Function to calculate the adjacency pairs coefficient
def adjacent_pairs_coefficient(sequence_list, amino_group, normalize=True):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure adjacency.
        normalize : bool (optional)
            Whether to normalize the results by the length
            of each sequence or not. Default is True.
    Output:
        Returns a list of values, indicating the adajcency
        pairs coefficient (or the count if normalize=False)
        from the given group in each input sequence.
    '''
    # Init results
    res = []

    # Operate for each sequence
    for s in sequence_list:
        # Build binary vector
        seq_vec = np.array(list(s))
        v = 1*np.isin(seq_vec, amino_group)

        # Build n+1 vector
        v_next = np.zeros(v.shape)
        v_next[1:] += v[:-1]

        # Compute adjacent pairs
        A = np.sum(v * v_next)

        # Normalize and return
        A_norm = A / (np.sum(v) - 1)

        if not normalize:
            res.append(A)
        else:
            res.append(A_norm)

    return res


# Function to calculate the adjacency triplets coefficient
def adjacent_triplets_coefficient(sequence_list, amino_group, normalize=True):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure adjacency.
        normalize : bool (optional)
            Whether to normalize the results by the length
            of each sequence or not. Default is True.
    Output:
        Returns a list of values, indicating the adajcency
        triplets coefficient (or the count if normalize=False)
        from the given group in each input sequence.
    '''
    # Init results
    res = []

    # Operate for each sequence
    for s in sequence_list:
        # Build binary vector
        seq_vec = np.array(list(s))
        v = 1*np.isin(seq_vec, amino_group)

        # Build n+1 and n+2 vectors
        v_next = np.zeros(v.shape)
        v_next[1:] += v[:-1]

        v_trp = np.zeros(v.shape)
        v_trp[2:] += v[:-2]

        # Compute adjacent triplets
        A = np.sum(v * v_next * v_trp)

        # Normalize and return
        A_norm = A / (np.sum(v) - 2)

        if not normalize:
            res.append(A)
        else:
            res.append(A_norm)

    return res


# Function to calculate the adjacency quadruplets coefficient
def adjacent_quadruplets_coefficient(sequence_list, amino_group, normalize=True):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure adjacency.
        normalize : bool (optional)
            Whether to normalize the results by the length
            of each sequence or not. Default is True.
    Output:
        Returns a list of values, indicating the adajcency
        quadruplets coefficient (or the count if normalize=False)
        from the given group in each input sequence.
    '''
    # Init results
    res = []

    # Operate for each sequence
    for s in sequence_list:
        # Build binary vector
        seq_vec = np.array(list(s))
        v = 1*np.isin(seq_vec, amino_group)

        # Build n+1 and n+2 vectors
        v_next = np.zeros(v.shape)
        v_next[1:] += v[:-1]

        v_trp = np.zeros(v.shape)
        v_trp[2:] += v[:-2]

        v_quad = np.zeros(v.shape)
        v_quad[3:] += v[:-3]

        # Compute adjacent triplets
        A = np.sum(v * v_next * v_trp * v_quad)

        # Normalize and return
        A_norm = A / max([(np.sum(v) - 3),1])

        if not normalize:
            res.append(A)
        else:
            res.append(A_norm)

    return res



# ----- OTHER ------
# Typical grouping of the standard amino acids
# Define amino acid groups
amino_groups_standard = {'Hydrophobic': ['C', 'M', 'V', 'L', 'I'],
 'Aromatic': ['W', 'Y', 'F'],
 'Hydrophilic': ['H', 'T', 'N', 'Q'],
 'Charged+': ['K', 'R'],
 'Charged-': ['D', 'E'],
 'Disorder promoting': ['A', 'S', 'G', 'P']}