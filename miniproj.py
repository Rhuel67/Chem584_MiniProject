import numpy as np
import math

np.set_printoptions(precision=2, linewidth= 150)

def kd(a:int, b:int) -> int:
    """
    Kronecker delta of two inputs.

    :param a: input 1
    :param 2: input 2
    """
    return int(a==b)

def inprod(m1:int, m2:int, n1:int, n2:int, s1:int=1, s2:int=1) -> int:
    """
    Returns the inner product <m1 m2, s1|n1 n2, s2>.
    """
    if (s1!=s2): return 0
    if (m1==m2) and (n1==n2): return kd(m1,n1)
    if (m1==m2) and (n1!=n2): return 0
    if (m1!=m2) and (n1==n2): return 0
    return kd(m1,n1)*kd(m2,n2) + s1*kd(m1,n2)*kd(m2,n1)

def elemA(m1:int, m2:int, n1:int, n2:int, s:int) -> float:
    '''
    Returns the matrix element of operator A given by <m1 m2, s1|A|n1 n2, s2>.
    '''
    if(n1==n2): return inprod(m1,m2,n1+1,n1-1)*math.sqrt(2*n1*(n1+1))
    if(n1==n2+2): 
        n = n2+1
        return inprod(m1,m2,n,n,s,1)*kd(s,1)*math.sqrt(2*n*(n+1)) + inprod(m1,m2,n+2,n-2,s,s)*math.sqrt((n-1)*(n+2))
    return inprod(m1,m2,n1+1,n2-1,s,s)*math.sqrt((n1+1)*n2) + inprod(m1,m2,n1-1,n2+1,s,s)*math.sqrt(n1*(n2+1))

def toinvcm(e:float) -> float:
    '''
    Converts an energy in Hartree to cm^-1.
    '''
    return e*219474.63136320

def tohart(e:float) -> float:
    '''
    Converts an energy in cm^-1 to Hartree.
    '''
    return e/219474.63136320

def get_e_n(om_e:float, om_x:float, n:int) -> float:
    '''
    Returns the energy of Morse oscillator eigenstate |n> given the frequency omega_e and 
    correction omega_x.

    :param om_e: oscillator frequency omega_e
    :param om_x: correction frequency omega_x
    :param n: energy level
    '''
    return om_e*(n+1/2) - om_x*(n+1/2)**2

def get_eig_func(vector:list[float], basis:list[tuple], sym:str) -> str: 
    ''' 
    Returns a string with the expression of an eigenstate as a linear combination of kets.
    Basis vectors are sorted by corresponding |c_i| and states with contribution |c_i|<10^5 
    are excluded.

    :param vector: list of c_i's
    :param basis: basis as a list of tuples of the form (n1,n2)
    :param sym: symmetry label of the basis (either '+' or '-')
    '''
    v = zip(vector,basis)
    vs = sorted(v, key=(lambda vi: vi[0]**2), reverse=True)
    ef_l = []
    for vsi in vs:
        if(abs(vsi[0])>=1E-5):
            ef_l.append((round(vsi[0],5), vsi[1][0], vsi[1][1]))
    
    if ef_l[0][0]>0:
        efunction = f" {ef_l[0][0]:1.5f}|{ef_l[0][1]} {ef_l[0][2]},{sym}>"
    else:
        efunction = f"-{-ef_l[0][0]:1.5f}|{ef_l[0][1]} {ef_l[0][2]},{sym}>"

    for i in range(1, len(ef_l)):
        if ef_l[i][0]>0:
            efunction += f" + {ef_l[i][0]:1.5f}|{ef_l[i][1]} {ef_l[i][2]},{sym}>"
        else:
            efunction += f" - {-ef_l[i][0]:1.5f}|{ef_l[i][1]} {ef_l[i][2]},{sym}>"

    return efunction

def get_eig_func_tex(vector:list[float], basis:list[tuple], sym:str) -> str: 
    ''' 
    Returns a string with the expression of an eigenstate as a linear combination of kets in 
    the format of the braket Latex package. Basis vectors are sorted by corresponding |c_i| 
    and states with contribution |c_i|<10^5 are excluded.

    :param vector: list of c_i's
    :param basis: basis as a list of tuples of the form (n1,n2)
    :param sym: symmetry label of the basis (either '+' or '-')
    '''
    v = zip(vector,basis)
    vs = sorted(v, key=(lambda vi: vi[0]**2), reverse=True)
    ef_l = []
    for vsi in vs:
        if(abs(vsi[0])>=1E-5):
            ef_l.append((round(vsi[0],5), vsi[1][0], vsi[1][1]))
    
    if ef_l[0][0]>0:
        efunction = f"{ef_l[0][0]:1.5f}" + r"\ket{" + f"{ef_l[0][1]}\\space{ef_l[0][2]},{sym}" + r"}"
    else:
        efunction = f"-{-ef_l[0][0]:1.5f}"+ r"\ket{" + f"{ef_l[0][1]}\\space{ef_l[0][2]},{sym}" + r"}"

    for i in range(1, len(ef_l)):
        if ef_l[i][0]>0:
            efunction += f"+{ef_l[i][0]:1.5f}"+ r"\ket{" + f"{ef_l[i][1]}\\space{ef_l[i][2]},{sym}" + r"}"
        else:
            efunction += f"-{-ef_l[i][0]:1.5f}" r"\ket{" + f"{ef_l[i][1]}\\space{ef_l[i][2]},{sym}" + r"}"

    return "$"+efunction+"$"

def calc_SA(basis: tuple[list[tuple[int, int]], list[tuple[int, int]], list[float]]):
    n1n2s_s = basis[0] # symmetric combined basis state labels, list of tuples (n1,n2) 
    n1n2s_a = basis[1] # antisymmetric combinded basis state labels, list of tuples (n1,n2) 
    
    N_size_s = len(n1n2s_s) # number of symmetric basis states
    N_size_a = len(n1n2s_a) # number of antisymmetric basis states
    
    S_s = np.zeros((N_size_s,N_size_s)) # matrix S for symmetric basis states
    A_s = np.zeros((N_size_s,N_size_s)) # matrix A for symmetric basis states

    S_a = np.zeros((N_size_a,N_size_a)) # matrix S for antisymmetric basis states
    A_a = np.zeros((N_size_a,N_size_a)) # matrix A for antisymmetric basis states

    # looping through all pairs of symmetric basis states to construct S and A
    for N1 in range(0,N_size_s):
        for N2 in range(0,N_size_s):
            m1 = n1n2s_s[N1][0]
            m2 = n1n2s_s[N1][1]
            n1 = n1n2s_s[N2][0]
            n2 = n1n2s_s[N2][1]

            S_s[N1][N2] = kd(m1,n1+1)*kd(m2,n2+1)*math.sqrt((n1+1)*(n2+1)) + (kd(m1,n1-1)*kd(m2,n2-1))*math.sqrt(n1*n2)
            A_s[N1][N2] = elemA(m1,m2,n1,n2,1)

    # looping through all pairs of antisymmetric basis states to construct S and A
    for N1 in range(0,N_size_a):
        for N2 in range(0,N_size_a):
            m1 = n1n2s_a[N1][0]
            m2 = n1n2s_a[N1][1]
            n1 = n1n2s_a[N2][0]
            n2 = n1n2s_a[N2][1]

            S_a[N1][N2] = kd(m1,n1+1)*kd(m2,n2+1)*math.sqrt((n1+1)*(n2+1)) + (kd(m1,n1-1)*kd(m2,n2-1))*math.sqrt(n1*n2)
            A_a[N1][N2] = elemA(m1,m2,n1,n2,-1)
    return ((S_s, S_a),(A_s, A_a))

def generate_basis_from_maxn(om_e:float, om_x:float, maxn: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[float]]:
    e_n = [] # list of energies of independent Morse oscillator eigenstates <= cutoff
    for n in range(0, maxn+1):
        e_n.append(get_e_n(om_e, om_x, n))

    n1n2s_s = [] # symmetric combined basis state labels, list of tuples (n1,n2) 
    n1n2s_a = [] # antisymmetric combinded basis state labels, list of tuples (n1,n2) 

    for Nt in range(0, maxn+1):
        for n2 in range(0, int((Nt+1)/2)):
            n1n2s_s.append((Nt-n2, n2))
            n1n2s_a.append((Nt-n2, n2))
        if (Nt%2==0): n1n2s_s.append((Nt-int(Nt/2), int(Nt/2)))
    
    return (n1n2s_s, n1n2s_a, e_n)

def generate_basis_from_cutoff(om_e:float, om_x:float, basis_cutoff:float) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[float]]:
    '''
    Returns 3-tuple containing a list of symmetric basis states, a list of antisymmetric 
    basis states, and the energies of the independent Morse oscillator states that make 
    up these bases. Basis states are <= cutoff energy. 

    :param om_e: oscillator frequency omega_e
    :param om_x: correction frequency omega_x
    :param basis_cutoff: the cutoff energy for the independent Morse oscillator basis states
    :return: a tuple of the form (symmetric basis, antisymmetric basis, oscillator energies)
    '''
    n=0
    e_n = [] # list of energies of independent Morse oscillator eigenstates <= cutoff
    while True: 
        e = get_e_n(om_e, om_x, n)
        if e>basis_cutoff: break
        e_n.append(e)
        n+=1

    Nt_max = n-1 # cutoff for n1+n2 as determined by basis_cutoff

    n1n2s_s = [] # symmetric combined basis state labels, list of tuples (n1,n2) 
    n1n2s_a = [] # antisymmetric combinded basis state labels, list of tuples (n1,n2) 

    for Nt in range(0, Nt_max+1):
        for n2 in range(0, int((Nt+1)/2)):
            n1n2s_s.append((Nt-n2, n2))
            n1n2s_a.append((Nt-n2, n2))
        if (Nt%2==0): n1n2s_s.append((Nt-int(Nt/2), int(Nt/2)))
    
    return (n1n2s_s, n1n2s_a, e_n)

def find_vib_levels_unknown_basis(om_e:float, om_x:float, 
                    c_S:float, c_A:float, 
                    basis_cutoff:float,
                    S:tuple | None=None, 
                    A:tuple | None=None):
    '''
    Generates a suitable basis and evaluates the eigenstates of H2D as a list of dicts. 

    :param om_e: oscillator frequency omega_e
    :param om_x: correction frequency omega_x
    :param c_S: the coefficient of operator S in the total Hamiltonian
    :param c_A: the coefficient of operator A in the total Hamiltonian
    :param basis_cutoff: the cutoff energy for the independent Morse oscillator basis states
    :param S: optional precalculated S matricies (S_symmetric, S_antisymmetric)
    :param A: optional precalculated A matricies (A_symmetric, A_antisymmetric)
    '''
    basis = generate_basis_from_cutoff(om_e, om_x, basis_cutoff)
    return find_vib_levels_fixed_basis(c_S, c_A, basis, S, A)
    
def find_vib_levels_fixed_basis(c_S:float, c_A:float, 
                    basis: tuple[list[tuple[int, int]], list[tuple[int, int]], list[float]],
                    S:tuple | None=None, 
                    A:tuple | None=None) -> list[dict]:
    '''
    Returns the eigenstates of H2D in a fixed basis as a list of dicts. 

    :param c_S: the coefficient of operator S in the total Hamiltonian
    :param c_A: the coefficient of operator A in the total Hamiltonian
    :param basis: a 3-tuple containing symmetric and antisymmetric basis states as well as 
    the energies of the independent Morse oscillator states that make up these bases
    :param S: optional precalculated S matricies (S_symmetric, S_antisymmetric)
    :param A: optional precalculated A matricies (A_symmetric, A_antisymmetric)
    '''

    e_n = basis[2] # list of energies of independent Morse oscillator eigenstates <= cutoff
    n1n2s_s = basis[0] # symmetric combined basis state labels, list of tuples (n1,n2) 
    n1n2s_a = basis[1] # antisymmetric combinded basis state labels, list of tuples (n1,n2) 

    e_N_s = [] # e_n1 + e_n2 for symmetric states
    for Nt in n1n2s_s:
        e_N_s.append(e_n[Nt[0]] + e_n[Nt[1]])

    e_N_a = [] # e_n1 + e_n2 for antisymmetric states
    for Nt in n1n2s_a:
        e_N_a.append(e_n[Nt[0]] + e_n[Nt[1]])

    N_size_s = len(n1n2s_s) # number of symmetric basis states
    N_size_a = len(n1n2s_a) # number of antisymmetric basis states


    H0_s = np.zeros((N_size_s,N_size_s)) # matrix H0 for symmetric basis states
    H0_a = np.zeros((N_size_a,N_size_a)) # matrix H0 for antisymmetric basis states

    # filling diagonal elements of H0 matricies with (e_n1 + en_2)
    np.fill_diagonal(H0_s, e_N_s)
    np.fill_diagonal(H0_a, e_N_a)

    # collecting elements of A and S in each basis
    if (S is None) or (A is None):
        ((S_s, S_a), (A_s, A_a)) = calc_SA(basis)
    else:
        (S_s, S_a) = S
        (A_s, A_a) = A
    
    # constructing total Hamiltonian matricies H2D = H0 + c_S*S + c_A*A
    H_s = H0_s + c_S*S_s + c_A*A_s
    H_a = H0_a + c_S*S_a + c_A*A_a
    
    # finding eigenvectors and eigenvalues of total Hamiltonian for symmetric states
    eig_s = np.linalg.eig(H_s)
    eigvals_s = eig_s[0]
    eigvecs_s = eig_s[1].T

    zpe = eigvals_s[0] # zero point energy (eigenvalue of state labeled (0,0,+))
    eoftrans_s = eigvals_s - zpe # subtracting zpe to find transition energies
    N_of_max_s = [np.argmax(np.square(vec)) for vec in eigvecs_s] # index of maximum contribution state, used to lookup n1 and n2 of that state

    # finding eigenvectors and eigenvalues of total Hamiltonian for antisymmetric states
    eig_a = np.linalg.eig(H_a)
    eigvals_a = eig_a[0]
    eigvecs_a = eig_a[1].T

    eoftrans_a = eigvals_a - zpe
    N_of_max_a = [np.argmax(np.square(vec)) for vec in eigvecs_a]

    result = [] # list of dicts containing information about all eigenstates

    # adding all symmetric eigenstates to result list
    for Nt in range(0, len(eoftrans_s)):
        result.append({
            "E": eigvals_s[Nt],
            "Et": eoftrans_s[Nt],
            "vector": eigvecs_s[Nt],
            "n1": n1n2s_s[N_of_max_s[Nt]][0], 
            "n2": n1n2s_s[N_of_max_s[Nt]][1], 
            "s": "+", 
            "efunction": get_eig_func(eigvecs_s[Nt], n1n2s_s, "+"),
            "efunction_tex": get_eig_func_tex(eigvecs_s[Nt], n1n2s_s, "+")
            })
        
    # adding all antisymmetric eigenstates to result list
    for Nt in range(0, len(eoftrans_a)):
        result.append({
            "E": eigvals_a[Nt],
            "Et": eoftrans_a[Nt],
            "vector": eigvecs_a[Nt],
            "n1": n1n2s_a[N_of_max_a[Nt]][0], 
            "n2": n1n2s_a[N_of_max_a[Nt]][1], 
            "s": "-", 
            "efunction": get_eig_func(eigvecs_a[Nt], n1n2s_a, "-"),
            "efunction_tex": get_eig_func_tex(eigvecs_s[Nt], n1n2s_s, "-")
            })

    # sorting result list by eigenvalue
    sorted_result = sorted(result, key=(lambda state: state["E"]))

    return sorted_result

if __name__=="__main__":
    om_e = 1.76361347017282E-02 # omega_e in atomic units
    om_x = 3.71826581833116E-04 # omega_x in atomic units

    c_S =  3.96513823698E-05 # c_S in atomic units
    c_A = -2.2440142339E-04 # c_a in atomic units

    basis = generate_basis_from_maxn(om_e,om_x,7)
    (S, A) = calc_SA(basis)

    sorted_result = find_vib_levels_fixed_basis(c_S, c_A, basis, S, A)
    
    # readable print format
    for state in sorted_result:
        print(f"({state['n1']},{state['n2']},{state['s']})")
        print(f"Transition energy (cm^-1): {toinvcm(state['Et'])}")
        print(f"Eigenfunction: {state['efunction']}")
        print()

    # parsable print format
    delim = '&'
    for state in sorted_result:
        out = delim.join([str(state["n1"]),str(state["n2"]),state["s"],f"{toinvcm(state['Et']):9.3f}",state["efunction_tex"]]) + r"\\\hline"
        print(out)

    
    

    