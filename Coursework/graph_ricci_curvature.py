import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple
import networkx as nx
import numpy as np
from scipy.optimize import linprog
import warnings

class GraphRicciCurvature:
    def __init__(self, G: nx.Graph):
        """
        Initialize the Ricci curvature calculator for a graph
        
        Args:
            G: NetworkX graph object
        """
        self.G = G
        self._cache = {}  # Cache for computed curvatures
        
    def _get_neighbors_distribution(self, node: int) -> Dict[int, float]:
        """
        Get the probability distribution over neighbors of a node.
        
        Args:
            node: The node to get distribution for
            
        Returns:
            Dictionary mapping neighbor nodes to probabilities
        """
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return {}
        
        # Uniform distribution over neighbors
        prob = 1.0 / len(neighbors)
        return {n: prob for n in neighbors}

    def node_wasserstein_distance(self, source_node, target_node, beta=1.0):
        """
        Calculate the Wasserstein distance between two nodes in a graph based on their
        heat kernel diffusion distributions.
        
        Parameters:
        -----------
        G : networkx.Graph
            The input graph
        source_node : node
            First node in the graph
        target_node : node
            Second node in the graph
        beta : float, optional (default=1.0)
            Diffusion time parameter for the heat kernel
            
        Returns:
        --------
        float
            The Wasserstein distance between the probability distributions
            centered at the two nodes
        """
        G = self.G

        # Verify nodes exist in the graph
        if source_node not in G or target_node not in G:
            raise ValueError("Source or target node not found in graph")
        
        # Get the adjacency matrix and number of nodes
        A = nx.adjacency_matrix(G).toarray()
        n = len(G.nodes())
        
        # Calculate the graph Laplacian
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        
        # Calculate heat kernel
        H = np.exp(-beta * L)
        
        # Get probability distributions for both nodes
        node_list = list(G.nodes())
        source_idx = node_list.index(source_node)
        target_idx = node_list.index(target_node)
        
        p = H[source_idx] / np.sum(H[source_idx])  # Source distribution
        q = H[target_idx] / np.sum(H[target_idx])  # Target distribution
        
        # Calculate shortest path distances between all pairs of nodes
        dist_matrix = dict(nx.all_pairs_shortest_path_length(G))
        distances = np.zeros((n, n))
        for i, node1 in enumerate(node_list):
            for j, node2 in enumerate(node_list):
                distances[i,j] = dist_matrix[node1][node2]
        
        # Set up the linear programming problem for optimal transport
        n_variables = n * n  # Number of variables in the transport matrix
        
        # Objective: minimize sum(distances * transport_matrix)
        c = distances.flatten()
        
        # Constraints for row sum
        A_row = np.zeros((n, n_variables))
        for i in range(n):
            A_row[i, i*n:(i+1)*n] = 1
        
        # Constraints for column sum
        A_col = np.zeros((n, n_variables))
        for i in range(n):
            A_col[i, i::n] = 1
        
        # Combine constraints
        A_eq = np.vstack([A_row, A_col])
        b_eq = np.concatenate([p, q])
        
        # Bounds for variables (non-negative)
        bounds = [(0, None) for _ in range(n_variables)]
        
        # Solve the linear programming problem
        result = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        if result.success:
            return result.fun
        else:
            raise RuntimeError("Failed to compute Wasserstein distance")
    
    def _wasserstein_distance(self, dist1: Dict[int, float], 
                            dist2: Dict[int, float]) -> float:
        """
        Calculate the Wasserstein distance (earth mover's distance) between two distributions
        
        Args:
            dist1: First probability distribution
            dist2: Second probability distribution
            
        Returns:
            The Wasserstein distance
        """
        # Get all nodes that appear in either distribution
        nodes1 = set(dist1.keys())
        nodes2 = set(dist2.keys())
        all_nodes = list(nodes1.union(nodes2))
        n = len(all_nodes)
        
        # Create node index mapping
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Create the cost matrix
        cost_matrix = np.zeros((n, n))
        for i, n1 in enumerate(all_nodes):
            for j, n2 in enumerate(all_nodes):
                if n1 == n2:
                    cost_matrix[i, j] = 0
                else:
                    try:
                        cost_matrix[i, j] = nx.shortest_path_length(self.G, n1, n2)
                    except nx.NetworkXNoPath:
                        cost_matrix[i, j] = float('inf')
        
        # Create supply arrays
        supply1 = np.zeros(n)
        supply2 = np.zeros(n)
        
        for node, prob in dist1.items():
            supply1[node_to_idx[node]] = prob
        for node, prob in dist2.items():
            supply2[node_to_idx[node]] = prob
            
        # Solve the optimal transport problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Calculate total cost
        total_cost = 0
        for i, j in zip(row_ind, col_ind):
            total_cost += cost_matrix[i, j] * min(supply1[i], supply2[j])
            
        return total_cost
    
    def compute_ricci_curvature(self, edge: Tuple[int, int]) -> float:
        """
        Compute the Ricci curvature for a given edge
        
        Args:
            edge: Tuple of (node1, node2) representing the edge
            
        Returns:
            The Ricci curvature value for the edge
        """
        if edge in self._cache:
            return self._cache[edge]
        
        node1, node2 = edge
        
        # # Get probability distributions
        # dist1 = self._get_neighbors_distribution(node1)
        # dist2 = self._get_neighbors_distribution(node2)
        # print(f"dist1 is:{dist1}")
        # print(f"dist2 is:{dist2}")


        # # Calculate Wasserstein distance
        # w_distance = self._wasserstein_distance(dist1, dist2)

        # New
        w_distance = self.node_wasserstein_distance(node1,node2)
        
        # Calculate Ricci curvature
        try:
            d = nx.shortest_path_length(self.G, node1, node2)
            curvature = 1 - (w_distance / d)
        except ZeroDivisionError:
            curvature = float('inf')
            
        self._cache[edge] = curvature
        return curvature
    
    def compute_ricci_curvatures(self) -> Dict[Tuple[int, int], float]:
        """
        Compute Ricci curvature for all edges in the graph
        
        Returns:
            Dictionary mapping edges to their Ricci curvature values
        """
        curvatures = {}
        for edge in self.G.edges():
            curvatures[edge] = self.compute_ricci_curvature(edge)
        return curvatures

# Example usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0),  # Square
        (1, 4), (2, 4)  # Additional edges
    ])
    
    # Initialize and compute Ricci curvatures
    ricci = GraphRicciCurvature(G)
    curvatures = ricci.compute_ricci_curvatures()
    
    # Print results
    for edge, curvature in curvatures.items():
        print(f"Edge {edge}: Ricci curvature = {curvature:.3f}")