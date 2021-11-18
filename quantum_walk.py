import retworkx
from retworkx.visualization import mpl_draw
import matplotlib.pyplot as plt
from qiskit import *
import numpy as np
from numpy import linalg as la
from scipy.linalg import expm
from qiskit.extensions import HamiltonianGate
from qiskit.visualization import plot_histogram
import imageio


def pad_zeros(adjacency):
    '''
    Helper function for padding zeros to increase adjacency
    matrix of shape (n,n) to (2**n, 2**n).

    Parameters: adjacency (ndarray): adjacency of graph
    Returns: full_matrix (ndarray): new adjacency with padded zeroes
    '''
    full_matrix = np.zeros((2 ** len(adjacency), 2 ** len(adjacency)))
    for i in range(len(adjacency)):
        for j in range(len(adjacency)):
            if adjacency[i][j] != 0:
                full_matrix[2 ** i][2 ** j] = adjacency[i][j]
    return full_matrix


def create_walk_circuit(adj_matrix, total_dur, num_snaps):
    '''
    Helper function for generating walk circuit with snapshots
    after each evolution of the quantum walk.

    Parameters: adj_matrix (ndarray):   adjacency of graph (2**n, 2**n)
                total_dur (float):      total time for quantum walk
                num_snaps (int):        number of snapshots throughout walk
    Returns:    circ (QuantumCircuit):  resulting circuit
    '''
    # create matrix exponential gate and circuit
    num_qubits = np.log2(len(adj_matrix))
    ExpGate = HamiltonianGate(adj_matrix, total_dur / num_snaps)
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits)
    circ = QuantumCircuit(qr, cr)

    # intialize to state |0...01> and add gate with snapshots
    circ.x(0)
    circ.snapshot(str(0))
    for i in range(num_snaps):
        circ.append(ExpGate, qr)
        circ.snapshot(str(i + 1))

    # return circuit
    return circ


def get_snapshots(adj_matrix, total_dur, num_snaps):
    '''
    Function for returning snapshots of quantum walk.

    Parameters: adj_matrix (ndarray):   adjacency of graph (2**n, 2**n)
                total_dur (float):      total time for quantum walk
                num_snaps (int):        number of snapshots throughout walk
    Returns:    map from iteration number to snapshot, snapshot counts
                up from binary in ordering (00, 01, 10, 11, ...)
    '''
    qc = create_walk_circuit(adj_matrix, total_dur, num_snaps)
    backend = Aer.get_backend('statevector_simulator')
    result = execute(qc, backend).result()
    return result.data()['snapshots']['statevector']


def generate_digraph_at_snapshot(adj_matrix, amplitude_array):
    '''
    Helper function that creates a graph for each snapshot.

    Parameters: adj_matrix (ndarray):       adjacency of graph (unpadded, nxn)
                amplitude_array (ndarray):  value from snapshot dictionary for a specific snapshot
    Returns: pydigraph and list of colors for each node in the graph
    '''
    g = retworkx.PyDiGraph()
    n = len(adj_matrix)

    # add nodes
    #lst = ["|" + str(bin(i))[2:].zfill(int(np.log2(n))) + ">" for i in range(n)]
    lst = ["|" + str(bin(2**i))[2:].zfill(int(n)) + ">" for i in range(n)]
    g.add_nodes_from(lst)

    # add edges
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[0])):
            if adj_matrix[i][j] != 0:
                g.add_edge(i, j, adj_matrix[i][j])

    # compute colors based on how probable the node is
    colors = []
    for i in range(len(adj_matrix)):
        alpha = abs(amplitude_array[2 ** i])

        # rescale our transparency
        alpha = alpha * 0.9 + 0.1
        colors.append((0.0, 0.0, 1.0, alpha))
    return g, colors


def generate_gif(adj_matrix, snapshots, gifname="quantum_walk", snapshot_dir="."):
    '''
    Function that makes a gif of the quantum walk.

    Parameters: adj_matrix (ndarray):       adjacency of graph (unpadded, nxn)
                snapshots (ndarray dict):   map from iteration number to snapshot, snapshot counts
                                            up from binary in ordering (00, 01, 10, 11 for 2 nodes)
                gifname (string):           name of the gif file created
                snapshot_dir (string):      name of the directory to store the snapshot png's
    Returns: saves a gif to the notebook files
    '''
    n = len(snapshots.items())
    pos = None
    # create all the images of the graphs
    for i in range(n):
        g, colors = generate_digraph_at_snapshot(adj_matrix, snapshots[str(i)][0])

        # save the position of the first graph so all subsequent graphs use the same node positioning
        if i == 0:
            pos = retworkx.spring_layout(g)

        plt.clf()
        mpl_draw(g, pos=pos, with_labels=True, labels=lambda node: node, arrows=False, node_size=1000, node_color= colors)
        plt.draw()
        plt.text(0.1, 0.1, 'snapshot ' + str(i), size=15, color='purple')
        plt.savefig(snapshot_dir + '/snapshot' + str(i) + '.png')

    # concatenate images into gif
    images = []
    filenames = [snapshot_dir + '/snapshot' + str(i) + '.png' for i in range(n)]
    for filename in filenames:
        images.append(imageio.imread(filename))
        imageio.mimsave(gifname + ".gif", images, duration = .5)


def visualize_walk(adj_matrix, total_dur, num_snaps, gifname="quantum_walk", snapshot_dir="."):
    '''
    Function for bringing it all together

    Parameters: adj_matrix (ndarray):   adjacency of graph (unpadded, nxn)
                total_dur (float):      total time for quantum walk
                num_snaps (int):        number of snapshots throughout walk
                gifname (string):           name of the gif file created
                snapshot_dir (string):      name of the directory to store the snapshot png's
    '''
    pad_adj = pad_zeros(adj_matrix)
    snaps = get_snapshots(pad_adj, total_dur, num_snaps)
    generate_gif(adj_matrix, snaps, gifname, snapshot_dir)
