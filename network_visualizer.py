from collections import defaultdict
import datetime
import random
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import logging
# import community

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def _chunk_betweenness(graph_data):
    """
    Standalone function for parallel betweenness calculation.
    Must be at module level for pickling to work.
    """
    nodes, edges = graph_data
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return nx.betweenness_centrality(G)

class NetworkVisualizer:
    def __init__(self, nodes_file, edges_file):
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        self.G = self.create_network()

    def create_network(self):
        """Create directed network from edge list"""
        logging.info("Creating network...")
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for _, row in self.nodes_df.iterrows():
            G.add_node(row['Id'], 
                      downloads=row['Downloads'],
                      version=row['Version'])
        
        # Add edges
        for _, row in self.edges_df.iterrows():
            G.add_edge(row['Source'], row['Target'])
        logging.info("Network created.")
        logging.info(f"Network created with nodes: {G.number_of_nodes()} and edges: {G.number_of_edges()}")
        return G

    def plot_network_structure(self, k_core=2, max_nodes=100, min_degree=20):
        """Plot network structure with better readability"""
        plt.figure(figsize=(20, 20), facecolor='white')
        
        # Create a copy and remove self-loops
        G_clean = self.G.copy()
        G_clean.remove_edges_from(nx.selfloop_edges(G_clean))
        
        # Filter to k-core and by in-degree
        core = nx.k_core(G_clean, k_core)
        in_degrees = dict(core.in_degree())
        
        # Get top nodes by in-degree
        significant_nodes = sorted([(n, d) for n, d in in_degrees.items()], 
                                key=lambda x: x[1], 
                                reverse=True)[:max_nodes]
        
        # Create subgraph
        subplot = core.subgraph([n[0] for n in significant_nodes])
        
        print(f"Plotting network with {len(subplot.nodes())} nodes and {len(subplot.edges())} edges")
        
        try:
            # Use Fruchterman-Reingold layout for better spread
            pos = nx.spring_layout(subplot, k=2, iterations=50)
            
            # Draw edges first with high transparency
            nx.draw_networkx_edges(subplot, pos,
                                edge_color='black',
                                alpha=0.2,
                                width=0.3)
            
            # Calculate node sizes based on in-degree
            max_degree = max(dict(subplot.in_degree()).values())
            node_sizes = [5000 * (subplot.in_degree(n) / max_degree) for n in subplot.nodes()]
            
            # Draw nodes with different sizes
            nx.draw_networkx_nodes(subplot, pos,
                                node_size=node_sizes,
                                node_color='lightblue',
                                alpha=0.6,
                                edgecolors='white')
            
            # Only label nodes with higher in-degree
            labels = {}
            for node in subplot.nodes():
                if subplot.in_degree(node) > (max_degree / 10):  # Only label top 20% nodes
                    labels[node] = node
            
            # Draw labels with white background for better readability
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            nx.draw_networkx_labels(subplot, pos,
                                labels,
                                font_size=8,
                                bbox=bbox_props)
            
            plt.title("PyPI Dependency Network - Most Depended Upon Packages")
            plt.axis('off')  # Turn off axis
            return plt.gcf()
            
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            return None
    def analyze_important_nodes(self):
        """Analyze node importance using PageRank and HITS"""
        print("\nAnalyzing important nodes...")
        
        # Calculate PageRank
        pagerank = nx.pagerank(self.G)
        
        # Calculate HITS
        hubs, authorities = nx.hits(self.G)
        
        # Combine scores
        node_scores = {}
        for node in self.G.nodes():
            node_scores[node] = {
                'pagerank': pagerank[node],
                'hub_score': hubs[node],
                'authority_score': authorities[node],
                'in_degree': self.G.in_degree(node),
                'out_degree': self.G.out_degree(node)
            }
        
        # Sort nodes by different metrics
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:10]
        top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\nTop 10 Packages by PageRank:")
        for pkg, score in top_pagerank:
            print(f"{pkg}: {score:.4f}")
            
        print("\nTop 10 Hub Packages:")
        for pkg, score in top_hubs:
            print(f"{pkg}: {score:.4f}")
            
        print("\nTop 10 Authority Packages:")
        for pkg, score in top_authorities:
            print(f"{pkg}: {score:.4f}")
            
        return node_scores

    def visualize_important_nodes(self):
        """Create visualizations for important nodes"""
        # Calculate metrics
        pagerank = nx.pagerank(self.G)
        hubs, authorities = nx.hits(self.G)
        
        plt.figure(figsize=(15, 5))
        
        # PageRank vs In-degree
        plt.subplot(131)
        plt.scatter([self.G.in_degree(n) for n in self.G.nodes()],
                    [pagerank[n] for n in self.G.nodes()],
                    alpha=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('In-degree')
        plt.ylabel('PageRank')
        plt.title('PageRank vs In-degree')
        
        # Hub scores vs Out-degree
        plt.subplot(132)
        plt.scatter([self.G.out_degree(n) for n in self.G.nodes()],
                    [hubs[n] for n in self.G.nodes()],
                    alpha=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Out-degree')
        plt.ylabel('Hub Score')
        plt.title('Hub Score vs Out-degree')
        
        # Authority scores vs In-degree
        plt.subplot(133)
        plt.scatter([self.G.in_degree(n) for n in self.G.nodes()],
                    [authorities[n] for n in self.G.nodes()],
                    alpha=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('In-degree')
        plt.ylabel('Authority Score')
        plt.title('Authority Score vs In-degree')
        
        plt.tight_layout()
        return plt.gcf()

    def analyze_assortativity(self):
        """Compute and visualize the assortativity of the graph."""
        # Compute assortativity coefficient
        assortativity = nx.degree_assortativity_coefficient(self.G)
        # assortativity = nx.degree_assortativity_coefficient(self.G, x='out', y='in')
        
        # Print the result
        print(f"Assortativity Coefficient: {assortativity:.4f}")
        
        # Visualize degree correlations (scatter plot)
        degrees = dict(self.G.degree())
        edge_degrees = [(degrees[u], degrees[v]) for u, v in self.G.edges()]
        x, y = zip(*edge_degrees)

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.5, s=10)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Degree of Node u')
        plt.ylabel('Degree of Node v')
        plt.title('Degree Correlation (Log-Log Scale)')
        plt.grid(True)
        plt.show()
        
        return assortativity

    def plot_rich_club(self, degree_type="in", n_random=10):
        """
        Plot rich-club coefficient vs degree for in-degree or out-degree,
        including comparison with random networks.
        
        Parameters:
            degree_type (str): "in" for in-degree, "out" for out-degree
            n_random (int): Number of random networks to generate for normalization
        """
        # Choose degree type
        if degree_type == "in":
            degree_func = self.G.in_degree
            title_degree = "In-Degree"
        elif degree_type == "out":
            degree_func = self.G.out_degree
            title_degree = "Out-Degree"
        else:
            raise ValueError("degree_type must be 'in' or 'out'.")

        # Calculate rich club coefficients for original network
        degrees = range(5, 50, 5)
        original_coeffs = []
        
        print("Calculating coefficients for original network...")
        for k in degrees:
            print(f"Calculating for {title_degree} > {k}")
            # Get nodes with degree > k
            high_degree_nodes = [n for n, d in degree_func() if d > k]
            
            if len(high_degree_nodes) < 2:
                original_coeffs.append(0)
                continue
            
            # Count edges between high-degree nodes
            edges_between = sum(1 for u, v in self.G.edges() 
                                if u in high_degree_nodes and v in high_degree_nodes)
            
            # Maximum possible edges (directed graph: n * (n-1))
            max_edges = len(high_degree_nodes) * (len(high_degree_nodes) - 1)
            
            # Compute rich-club coefficient
            original_coeffs.append(edges_between / max_edges if max_edges > 0 else 0)

        # Generate random networks and calculate their coefficients
        print("\nCalculating coefficients for random networks...")
        random_coeffs = []
        for i in range(n_random):
            print(f"Processing random network {i+1}/{n_random}")
            
            # Generate random network preserving degree sequence
            in_seq = [d for n, d in self.G.in_degree()]
            out_seq = [d for n, d in self.G.out_degree()]
            random_G = nx.directed_configuration_model(in_seq, out_seq)
            
            # Calculate coefficients for random network
            rand_coeffs = []
            for k in degrees:
                if degree_type == "in":
                    rand_degree_func = random_G.in_degree
                else:
                    rand_degree_func = random_G.out_degree
                    
                high_degree_nodes = [n for n, d in rand_degree_func() if d > k]
                
                if len(high_degree_nodes) < 2:
                    rand_coeffs.append(0)
                    continue
                    
                edges_between = sum(1 for u, v in random_G.edges() 
                                if u in high_degree_nodes and v in high_degree_nodes)
                max_edges = len(high_degree_nodes) * (len(high_degree_nodes) - 1)
                rand_coeffs.append(edges_between / max_edges if max_edges > 0 else 0)
                
            random_coeffs.append(rand_coeffs)

        # Calculate average random coefficients
        avg_random_coeffs = np.mean(random_coeffs, axis=0)
        
        # Calculate normalized coefficients
        normalized_coeffs = [orig / rand if rand > 0 else 0 
                            for orig, rand in zip(original_coeffs, avg_random_coeffs)]

        # Create subplots for both raw and normalized coefficients
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot raw coefficients
        ax1.plot(degrees[:len(original_coeffs)], original_coeffs, 'bo-', label='Original Network')
        ax1.plot(degrees[:len(avg_random_coeffs)], avg_random_coeffs, 'ro--', label='Random Networks')
        ax1.set_xscale('log')
        ax1.set_xlabel(f'{title_degree} Threshold (k)')
        ax1.set_ylabel('Rich Club Coefficient')
        ax1.set_title(f'Raw Rich Club Coefficient: {title_degree}')
        ax1.grid(True)
        ax1.legend()

        # Plot normalized coefficients
        ax2.plot(degrees[:len(normalized_coeffs)], normalized_coeffs, 'go-')
        ax2.axhline(y=1, color='r', linestyle='--', label='Random Baseline')
        ax2.set_xscale('log')
        ax2.set_xlabel(f'{title_degree} Threshold (k)')
        ax2.set_ylabel('Normalized Rich Club Coefficient')
        ax2.set_title(f'Normalized Rich Club Coefficient: {title_degree}')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        return plt.gcf(), {
            'degrees': degrees,
            'original_coeffs': original_coeffs,
            'random_coeffs': avg_random_coeffs,
            'normalized_coeffs': normalized_coeffs
        }

    def analyze_rich_club_composition(self, degree_type="in", k_threshold=20):
        """
        Analyze the composition of the rich club at a given threshold.
        
        Parameters:
            degree_type (str): "in" for in-degree, "out" for out-degree
            k_threshold (int): Degree threshold for rich club membership
        
        Returns:
            dict: Metrics about the rich club composition
        """
        # Choose degree type
        if degree_type == "in":
            degree_func = self.G.in_degree
            title_degree = "In-Degree"
        else:
            degree_func = self.G.out_degree
            title_degree = "Out-Degree"

        # Get rich club nodes
        rich_nodes = [n for n, d in degree_func() if d > k_threshold]
        
        if len(rich_nodes) < 2:
            return {
                'size': 0,
                'density': 0,
                'clustering': 0,
                'edge_count': 0,
                'avg_degree': 0
            }
        
        # Create subgraph of rich club
        rich_club = self.G.subgraph(rich_nodes)
        
        # Calculate metrics
        metrics = {
            'size': len(rich_nodes),
            'density': nx.density(rich_club),
            'clustering': nx.average_clustering(rich_club),
            'edge_count': rich_club.number_of_edges(),
            'avg_degree': sum(dict(rich_club.degree()).values()) / len(rich_nodes)
        }
        
        print(f"\nRich Club Composition Analysis ({title_degree} > {k_threshold}):")
        print(f"Number of nodes: {metrics['size']}")
        print(f"Density: {metrics['density']:.4f}")
        print(f"Average clustering coefficient: {metrics['clustering']:.4f}")
        print(f"Number of edges: {metrics['edge_count']}")
        print(f"Average degree: {metrics['avg_degree']:.2f}")
        
        return metrics


    def plot_betweenness_vs_degree(self):
        """Plot betweenness centrality vs degree using sampling"""
        import random
        print("Calculating approximate betweenness centrality...")
        
        # Sample nodes for betweenness calculation
        # For a very large network, we'll use an even smaller sample
        n_samples = min(1000, len(self.G.nodes()))
        sampled_nodes = random.sample(list(self.G.nodes()), n_samples)
        
        # Calculate betweenness only for sampled nodes
        betweenness = nx.betweenness_centrality(
            self.G, 
            k=100,  # Use sampling parameter
            normalized=True,
            endpoints=False,
            seed=42
        )
        
        print("Calculating degree...")
        degrees = dict(self.G.degree())
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        
        # Convert to lists for plotting
        degree_values = list(degrees.values())
        betweenness_values = list(betweenness.values())
        
        plt.scatter(degree_values, 
                betweenness_values,
                alpha=0.5,
                s=20)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Degree')
        plt.ylabel('Betweenness Centrality')
        plt.title('Betweenness Centrality vs Degree (Sampled)')
        plt.grid(True)
        
        # Annotate only the top nodes by betweenness
        top_nodes = sorted(
            betweenness.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        for node, b_cent in top_nodes:
            plt.annotate(
                node, 
                (degrees[node], b_cent),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8
            )
        
        return plt.gcf()

    def plot_degree_distribution(self):
        def power_law(x, a, b):
            """Power law function for curve fitting."""
            return a * np.power(x, b)
        """Plot in-degree and out-degree distributions."""
        print("Plotting degree distributions...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # In-degree
        in_degrees = [d for n, d in self.G.in_degree()]
        in_degrees = [d for d in in_degrees if d > 0]  # Remove zeros
        unique_degrees, counts = np.unique(in_degrees, return_counts=True)
        log_degrees = np.log10(unique_degrees)
        log_counts = np.log10(counts)
        
        ax1.scatter(unique_degrees, counts, s=10, alpha=0.7, label="Data")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title('In-Degree Distribution')
        ax1.set_xlabel('Degree (log)')
        ax1.set_ylabel('Count (log)')
        
        # Fit a power law to in-degree
        if len(unique_degrees) > 0:
            popt, _ = curve_fit(lambda x, a, b: a * np.power(x, b), unique_degrees, counts, maxfev=10000)
            x_fit = np.linspace(min(unique_degrees), max(unique_degrees), 100)
            ax1.plot(x_fit, power_law(x_fit, *popt), 'r--', label=f'Power law fit (a={popt[0]:.2f}, b={popt[1]:.2f})')
            ax1.legend()

        # Out-degree
        out_degrees = [d for n, d in self.G.out_degree()]
        out_degrees = [d for d in out_degrees if d > 0]  # Remove zeros
        unique_degrees, counts = np.unique(out_degrees, return_counts=True)
        
        ax2.scatter(unique_degrees, counts, s=10, alpha=0.7, label="Data")
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_title('Out-Degree Distribution')
        ax2.set_xlabel('Degree (log)')
        ax2.set_ylabel('Count (log)')
        
        # Fit a power law to out-degree
        if len(unique_degrees) > 0:
            popt, _ = curve_fit(lambda x, a, b: a * np.power(x, b), unique_degrees, counts, maxfev=10000)
            x_fit = np.linspace(min(unique_degrees), max(unique_degrees), 100)
            ax2.plot(x_fit, power_law(x_fit, *popt), 'r--', label=f'Power law fit (a={popt[0]:.2f}, b={popt[1]:.2f})')
            ax2.legend()

        plt.tight_layout()
        print("Degree distributions plotted as scatter plots with power-law fits.")
        return plt.gcf()

    def print_network_stats(self):
        """Print basic network statistics"""
        logging.info(f"\nNetwork Statistics:")
        logging.info(f"Number of nodes: {self.G.number_of_nodes()}")
        logging.info(f"Number of edges: {self.G.number_of_edges()}")
        logging.info(f"Number of self-loops: {len(list(nx.selfloop_edges(self.G)))}")
        
        # Check degree statistics
        in_degrees = [d for n, d in self.G.in_degree()]
        out_degrees = [d for n, d in self.G.out_degree()]
        
        logging.info(f"\nDegree Statistics:")
        logging.info(f"Max in-degree: {max(in_degrees)}")
        logging.info(f"Max out-degree: {max(out_degrees)}")
        logging.info(f"Average in-degree: {sum(in_degrees)/len(in_degrees):.2f}")
        logging.info(f"Average out-degree: {sum(out_degrees)/len(out_degrees):.2f}")

    def simulate_cascade_failure(self, n_remove=10, strategy="pagerank"):
        """
        Simulate cascade failure by removing top nodes based on a strategy
        and measure the impact on the network.
        
        Parameters:
        - n_remove: Number of nodes to remove.
        - strategy: "pagerank", "hubs", "authorities", or "random".
        
        Returns:
        - results: A list of dictionaries with the impact metrics at each step.
        """
        logging.info(f"Simulating cascade failure ({strategy})...")
        G_copy = self.G.copy()

        # Compute centrality metrics
        logging.info("Computing centrality metrics...")
        if strategy == "pagerank":
            pagerank = nx.pagerank(G_copy)
            sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        elif strategy == "hubs":
            hubs, authorities = nx.hits(G_copy)
            sorted_nodes = sorted(hubs.items(), key=lambda x: x[1], reverse=True)
        elif strategy == "authorities":
            hubs, authorities = nx.hits(G_copy)
            sorted_nodes = sorted(authorities.items(), key=lambda x: x[1], reverse=True)
        elif strategy == "random":
            sorted_nodes = list(G_copy.nodes())  # Just the node IDs for random removal
            np.random.shuffle(sorted_nodes)
        else:
            raise ValueError("Invalid strategy. Choose from 'pagerank', 'hubs', 'authorities', 'random'.")

        # Track impact metrics
        results = []
        logging.info(f"Removing {n_remove} nodes based on {strategy} strategy.")
        for i in range(n_remove):
            if i % 100 == 0:
                logging.info(f"Removed node {i + 1}")
            if i < len(sorted_nodes):
                # Extract node ID (for non-random strategies, extract the first element of the tuple)
                node = sorted_nodes[i][0] if strategy != "random" else sorted_nodes[i]
                
                if node not in G_copy:
                    logging.info(f"Warning: Node {node} not found in the graph. Skipping.")
                    continue
                
                G_copy.remove_node(node)
                
                # Measure impact
                largest_cc = len(max(nx.strongly_connected_components(G_copy), key=len))
                num_components = nx.number_strongly_connected_components(G_copy)
                remaining_edges = G_copy.number_of_edges()

                results.append({
                    "nodes_removed": i + 1,
                    "largest_component_size": largest_cc,
                    "num_components": num_components,
                    "remaining_edges": remaining_edges
                })
        logging.info("Simulation completed.")
        return results
    
    def simulate_cascade_failure_weak(self, n_remove=2502, strategy="pagerank"):
        """
        Simulate cascade failure by removing top nodes based on a strategy
        and measure the impact on the network.
        
        Parameters:
        - n_remove: Number of nodes to remove (default 2502 based on rich club threshold)
        - strategy: "pagerank", "hubs", "authorities", or "random"
        
        Returns:
        - results: List of dictionaries with impact metrics at each step
        """
        print(f"Simulating cascade failure ({strategy})...")
        G_copy = self.G.copy()
        initial_nodes = G_copy.number_of_nodes()
        initial_edges = G_copy.number_of_edges()

        # Compute centrality metrics based on strategy
        if strategy == "pagerank":
            centrality = nx.pagerank(G_copy)
        elif strategy == "hubs":
            hubs, _ = nx.hits(G_copy)
            centrality = hubs
        elif strategy == "authorities":
            _, authorities = nx.hits(G_copy)
            centrality = authorities
        elif strategy == "random":
            nodes = list(G_copy.nodes())
            np.random.shuffle(nodes)
            centrality = {node: 1 for node in nodes}  # Dummy values for consistent processing
        else:
            raise ValueError("Invalid strategy. Choose from 'pagerank', 'hubs', 'authorities', 'random'")

        # Sort nodes by centrality
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        results = []

        for i in range(n_remove):
            if i >= len(sorted_nodes):
                break
                
            node = sorted_nodes[i][0]
            if node not in G_copy:
                continue

            # Find all nodes that depend on the removed node (cascade effect)
            affected_nodes = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in affected_nodes:
                    affected_nodes.add(current)
                    # Add nodes that depend on current node to stack
                    stack.extend([n for n in G_copy.predecessors(current) 
                                if n not in affected_nodes])

            # Remove affected nodes
            G_copy.remove_nodes_from(affected_nodes)

            # Measure impact
            if G_copy.number_of_nodes() > 0:
                largest_wcc = len(max(nx.weakly_connected_components(G_copy), key=len))
                relative_size = largest_wcc / initial_nodes
                components = nx.number_weakly_connected_components(G_copy)
                remaining_edges = G_copy.number_of_edges()
                edge_density = remaining_edges / initial_edges if initial_edges > 0 else 0
            else:
                largest_wcc = 0
                relative_size = 0
                components = 0
                remaining_edges = 0
                edge_density = 0

            results.append({
                "nodes_removed": i + 1,
                "nodes_affected": len(affected_nodes),
                "largest_component_size": largest_wcc,
                "relative_size": relative_size,
                "num_components": components,
                "remaining_edges": remaining_edges,
                "edge_density": edge_density
            })

            if i % 100 == 0:
                print(f"Removed {i+1} nodes, affected {len(affected_nodes)} nodes")

        return results
    
    def simulate_random_edge_removal_node_degree(self, n_remove=2502):
        """
        Simulate cascading failure by removing edges randomly from nodes with in-degree > 0.
        
        Parameters:
            n_remove (int): Number of edges to remove (default matches node removal count)
        
        Returns:
            List of dictionaries containing impact metrics at each step
        """
        print("Simulating random edge removal...")
        G_copy = self.G.copy()
        initial_nodes = G_copy.number_of_nodes()
        initial_edges = G_copy.number_of_edges()
        
        # Only consider edges where target node has in-degree > 0
        valid_edges = [(u, v) for u, v in G_copy.edges() 
                    if G_copy.in_degree(v) > 0]
        np.random.shuffle(valid_edges)  # Randomize edge removal order

        results = []
        for i in range(min(n_remove, len(valid_edges))):
            # Remove the edge and identify affected nodes
            edge = valid_edges[i]
            source, target = edge
            G_copy.remove_edge(source, target)
            
            # Find nodes affected by this edge removal through dependency chain
            affected_nodes = set()
            stack = [target]  # Start with the target node of removed edge
            while stack:
                current = stack.pop()
                if current not in affected_nodes:
                    affected_nodes.add(current)
                    # Add nodes that depend on current node to stack
                    stack.extend([n for n in G_copy.predecessors(current) 
                                if n not in affected_nodes])

            # Measure impact
            if G_copy.number_of_nodes() > 0:
                largest_wcc = len(max(nx.weakly_connected_components(G_copy), key=len))
                relative_size = largest_wcc / initial_nodes
                components = nx.number_weakly_connected_components(G_copy)
                remaining_edges = G_copy.number_of_edges()
            else:
                largest_wcc = 0
                relative_size = 0
                components = 0
                remaining_edges = 0

            results.append({
                "edges_removed": i + 1,
                "nodes_affected": len(affected_nodes),
                "cascade_multiplier": len(affected_nodes) / (i + 1),
                "largest_component_size": largest_wcc,
                "relative_size": relative_size,
                "num_components": components,
                "remaining_edges": remaining_edges
            })

            if i % 100 == 0:
                print(f"Removed edge {i+1}, affected {len(affected_nodes)} nodes")

        print(f"Completed random edge removal simulation after removing {len(results)} edges")
        return results

    def simulate_cascade_failure_weak_node_degree(self, n_remove=2502, strategy="pagerank"):
        """
        Simulate cascade failure by removing nodes based on strategy, only considering
        nodes with in-degree > 0.
        
        Parameters:
        - n_remove: Number of nodes to remove
        - strategy: "pagerank", "hubs", "authorities", or "random"
        
        Returns:
        - results: List of dictionaries with impact metrics at each step
        """
        print(f"Simulating cascade failure ({strategy})...")
        G_copy = self.G.copy()
        initial_nodes = G_copy.number_of_nodes()
        initial_edges = G_copy.number_of_edges()

        # Get nodes with in-degree > 0
        valid_nodes = [n for n in G_copy.nodes() if G_copy.in_degree(n) > 0]
        
        # Create subgraph of valid nodes for centrality calculations
        valid_graph = G_copy.subgraph(valid_nodes)

        # Compute centrality metrics based on strategy
        if strategy == "pagerank":
            centrality = nx.pagerank(valid_graph)
        elif strategy == "hubs":
            hubs, _ = nx.hits(valid_graph)
            centrality = hubs
        elif strategy == "authorities":
            _, authorities = nx.hits(valid_graph)
            centrality = authorities
        elif strategy == "random":
            centrality = {node: random.random() for node in valid_nodes}
        else:
            raise ValueError("Invalid strategy")

        # Sort nodes by centrality
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        results = []

        for i in range(min(n_remove, len(sorted_nodes))):
            node = sorted_nodes[i][0]
            if node not in G_copy:
                continue

            # Find cascade effect
            affected_nodes = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in affected_nodes:
                    affected_nodes.add(current)
                    stack.extend([n for n in G_copy.predecessors(current) 
                                if n not in affected_nodes])

            # Remove affected nodes
            G_copy.remove_nodes_from(affected_nodes)

            # Measure impact
            if G_copy.number_of_nodes() > 0:
                largest_wcc = len(max(nx.weakly_connected_components(G_copy), key=len))
                relative_size = largest_wcc / initial_nodes
                components = nx.number_weakly_connected_components(G_copy)
                remaining_edges = G_copy.number_of_edges()
            else:
                largest_wcc = 0
                relative_size = 0
                components = 0
                remaining_edges = 0

            results.append({
                "nodes_removed": i + 1,
                "nodes_affected": len(affected_nodes),
                "largest_component_size": largest_wcc,
                "relative_size": relative_size,
                "num_components": components,
                "remaining_edges": remaining_edges
            })

            if i % 100 == 0:
                print(f"Removed {i+1} nodes, affected {len(affected_nodes)} nodes")

        return results
    
    def get_rich_club_nodes(self, degree_type="in", degree_threshold=10, normalize=True):
        """
        Identify nodes belonging to the rich club while calculating their influence 
        metrics to better understand their role in the network.
        
        Parameters:
            degree_type (str): "in" for in-degree, "out" for out-degree
            degree_threshold (int): Minimum degree for rich-club membership
            normalize (bool): Whether to normalize rich club coefficients
        
        Returns:
            List of tuples: (node, metrics) for rich club members
        """
        print(f"Analyzing rich club structure for {degree_type}-degree > {degree_threshold}...")
        
        # Set up degree function based on type
        if degree_type == "in":
            degree_func = self.G.in_degree
            neighbor_func = self.G.predecessors
        elif degree_type == "out":
            degree_func = self.G.out_degree
            neighbor_func = self.G.successors
        else:
            raise ValueError("degree_type must be 'in' or 'out'")

        # Identify rich club nodes and calculate their metrics
        rich_club_nodes = []
        degrees = dict(degree_func())
        
        for node, degree in degrees.items():
            if degree > degree_threshold:
                # Calculate node's influence metrics
                downstream_nodes = set()
                stack = [node]
                while stack:
                    current = stack.pop()
                    for successor in self.G.successors(current):
                        if successor not in downstream_nodes:
                            downstream_nodes.add(successor)
                            stack.append(successor)
                
                # Calculate local clustering coefficient
                local_clustering = nx.clustering(self.G, node)
                
                # Store node with its metrics
                rich_club_nodes.append({
                    'node': node,
                    'degree': degree,
                    'downstream_impact': len(downstream_nodes),
                    'clustering': local_clustering
                })
        
        # Sort by impact
        rich_club_nodes.sort(key=lambda x: x['downstream_impact'], reverse=True)
        
        print(f"Rich Club Analysis Results:")
        print(f"Found {len(rich_club_nodes)} rich-club nodes")
        print(f"Average downstream impact: {np.mean([n['downstream_impact'] for n in rich_club_nodes]):.2f}")
        print(f"Average clustering: {np.mean([n['clustering'] for n in rich_club_nodes]):.2f}")
        
        return rich_club_nodes

    def simulate_rich_club_failure_batched(self, rich_club_nodes, batch_size=10):
        """
        Simulate cascading failures by removing rich-club nodes in batches while tracking detailed metrics
        about network structure and cascade effects.
        
        Parameters:
            rich_club_nodes (list): List of dictionaries containing rich club nodes and their metrics
            batch_size (int): Number of nodes to process in each batch
        
        Returns:
            List of dictionaries containing impact metrics at each step
        """
        print("Simulating rich-club failure cascade in batches...")
        G_copy = self.G.copy()
        initial_nodes = G_copy.number_of_nodes()
        initial_edges = G_copy.number_of_edges()
        results = []
        total_nodes_removed = 0

        # Process nodes in batches
        for batch_start in range(0, len(rich_club_nodes), batch_size):
            # Get current batch of nodes
            batch = rich_club_nodes[batch_start:batch_start + batch_size]
            batch_affected_nodes = set()
            batch_metrics = {
                'total_downstream_impact': 0,
                'total_clustering': 0,
                'nodes_in_batch': len(batch)
            }
            
            # Process each node in the batch to calculate combined effects
            for node_data in batch:
                node = node_data['node']
                if node in G_copy:
                    # Track metrics for this node
                    batch_metrics['total_downstream_impact'] += node_data['downstream_impact']
                    batch_metrics['total_clustering'] += node_data['clustering']
                    
                    # Find cascade effect through dependency chain
                    stack = [node]
                    while stack:
                        current = stack.pop()
                        if current not in batch_affected_nodes:
                            batch_affected_nodes.add(current)
                            # Add nodes that depend on current node
                            stack.extend([n for n in G_copy.predecessors(current) 
                                        if n not in batch_affected_nodes])

            # Remove all affected nodes at once for this batch
            G_copy.remove_nodes_from(batch_affected_nodes)
            total_nodes_removed += len(batch)
            
            # Calculate network metrics after batch removal
            if G_copy.number_of_nodes() > 0:
                largest_wcc = len(max(nx.weakly_connected_components(G_copy), key=len))
                relative_size = largest_wcc / initial_nodes
                components = nx.number_weakly_connected_components(G_copy)
                remaining_edges = G_copy.number_of_edges()
            else:
                largest_wcc = 0
                relative_size = 0
                components = 0
                remaining_edges = 0

            # Store comprehensive metrics for this batch
            results.append({
                "nodes_removed": total_nodes_removed,
                "nodes_affected": len(batch_affected_nodes),
                "cascade_multiplier": len(batch_affected_nodes) / len(batch) if len(batch) > 0 else 0,
                "largest_component_size": largest_wcc,
                "relative_size": relative_size,
                "num_components": components,
                "remaining_edges": remaining_edges,
                "removed_node_impact": batch_metrics['total_downstream_impact'] / batch_metrics['nodes_in_batch'],
                "removed_node_clustering": batch_metrics['total_clustering'] / batch_metrics['nodes_in_batch']
            })

            print(f"Processed batch {batch_start//batch_size + 1}, "
                f"removed {total_nodes_removed} nodes total, "
                f"affected {len(batch_affected_nodes)} nodes in this batch")

        print("Rich club failure simulation completed")
        return results
    
    def plot_rich_club_cascade(self, results):
        """
        Visualize the impact of removing rich club nodes on network structure.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot our three core metrics
        plt.plot([r["nodes_removed"] for r in results], 
                [r["largest_component_size"] for r in results], 
                'b-', label='Largest Component Size')
        
        plt.plot([r["nodes_removed"] for r in results], 
                [r["num_components"] for r in results], 
                'r-', label='Number of Components')
        
        plt.plot([r["nodes_removed"] for r in results], 
                [r["remaining_edges"] for r in results], 
                'g-', label='Remaining Edges')
        
        plt.xlabel('Rich Club Nodes Removed')
        plt.ylabel('Impact Metrics')
        plt.title('Rich Club Cascade Impact')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
    
    def plot_rich_club_characteristics(self, results):
        """
        Visualize rich club specific metrics during cascade.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top plot: Cascade multiplication effect
        ax1.plot([r["nodes_removed"] for r in results],
                [r["cascade_multiplier"] for r in results],
                'b-', label='Cascade Multiplier')
        ax1.plot([r["nodes_removed"] for r in results],
                [r["relative_size"] for r in results],
                'r--', label='Relative Network Size')
        ax1.set_ylabel('Multiplier Effect')
        ax1.set_title('Rich Club Cascade Amplification')
        ax1.legend()
        ax1.grid(True)
        
        # Bottom plot: Node characteristics
        ax2.plot([r["nodes_removed"] for r in results],
                [r["removed_node_impact"] for r in results],
                'g-', label='Removed Node Impact')
        ax2.plot([r["nodes_removed"] for r in results],
                [r["removed_node_clustering"] for r in results],
                'purple', label='Local Clustering', linestyle='--')
        ax2.set_xlabel('Rich Club Nodes Removed')
        ax2.set_ylabel('Node Metrics')
        ax2.set_title('Rich Club Node Characteristics')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    def simulate_weak_tie_failure(self, n_remove=10):
        """
        Simulate cascading failure by removing nodes with the lowest degree.
        
        Parameters:
            n_remove (int): Number of nodes to remove.
        
        Returns:
            List of impact metrics after each removal.
        """
        print("Simulating weak-tie failure (removing lowest-degree nodes)...")
        G_copy = self.G.copy()
        results = []
        
        # Sort nodes by degree (ascending)
        low_degree_nodes = sorted(G_copy.degree, key=lambda x: x[1])[:n_remove]
        print(f"Removing {n_remove} nodes with lowest degree.")
        
        for i, (node, degree) in enumerate(low_degree_nodes):
            if node in G_copy:
                G_copy.remove_node(node)

                # Measure impact
                largest_cc = len(max(nx.strongly_connected_components(G_copy), key=len))
                num_components = nx.number_strongly_connected_components(G_copy)
                remaining_edges = G_copy.number_of_edges()

                results.append({
                    "nodes_removed": i + 1,
                    "largest_component_size": largest_cc,
                    "num_components": num_components,
                    "remaining_edges": remaining_edges
                })
            if i % 20 == 0:
                print(f"Removed node {i + 1} of {len(low_degree_nodes)}")
        
        print("Weak-tie failure simulation completed.")
        return results

    def get_stratified_low_degree_nodes(self, target_count=2502, max_in_degree=1):
        """
        Get low-degree nodes stratified by their out-degree.
        This helps identify potentially more important low-degree nodes.
        
        Parameters:
            target_count (int): Number of nodes to sample
            max_in_degree (int): Maximum in-degree to consider
        
        Returns:
            list: Strategically sampled low-degree nodes
        """
        # Get low degree nodes
        low_degree_nodes = [n for n, d in self.G.in_degree() if d <= max_in_degree]
        print(f"Found {len(low_degree_nodes)} nodes with in-degree <= {max_in_degree}")
        
        # Group nodes by out-degree
        out_degree_groups = defaultdict(list)  # noqa: F821
        for node in low_degree_nodes:
            out_degree = self.G.out_degree(node)
            out_degree_groups[out_degree].append(node)
        
        # Sample proportionally from each out-degree group, prioritizing higher out-degrees
        selected_nodes = []
        total_nodes = len(low_degree_nodes)
        print(f"Total nodes: {total_nodes}")
        for out_degree, nodes in sorted(out_degree_groups.items(), reverse=True):
            print(f"Out-degree: {out_degree}, Nodes: {len(nodes)}")
            # Calculate how many nodes to sample from this group
            group_size = len(nodes)
            sample_size = int((group_size / total_nodes) * target_count)
            
            if sample_size > 0:
                # Sample from this group
                selected = random.sample(nodes, min(sample_size, len(nodes)))
                selected_nodes.extend(selected)
        
        # Fill any remaining slots randomly
        remaining = target_count - len(selected_nodes)
        if remaining > 0:
            remaining_nodes = [n for n in low_degree_nodes if n not in selected_nodes]
            if remaining_nodes:
                selected_nodes.extend(random.sample(remaining_nodes, min(remaining, len(remaining_nodes))))
        
        print(f"Selected {len(selected_nodes)} nodes using stratified sampling")
        return selected_nodes[:target_count]
        
    def simulate_random_edge_removal(self, n_remove=2502):
        """
        Simulate cascading failure by removing edges randomly and tracking the cascade
        effects through the network's dependency structure.
        
        Parameters:
            n_remove (int): Number of edges to remove (default matches node removal count)
        
        Returns:
            List of dictionaries containing impact metrics at each step
        """
        print("Simulating random edge removal...")
        G_copy = self.G.copy()
        initial_nodes = G_copy.number_of_nodes()
        initial_edges = G_copy.number_of_edges()
        edges = list(G_copy.edges())
        np.random.shuffle(edges)  # Randomize edge removal order

        results = []
        for i in range(n_remove):
            if i >= len(edges):
                break
                
            # Remove the edge and identify affected nodes
            edge = edges[i]
            source, target = edge
            G_copy.remove_edge(source, target)
            
            # Find nodes affected by this edge removal through dependency chain
            affected_nodes = set()
            stack = [target]  # Start with the target node of removed edge
            while stack:
                current = stack.pop()
                if current not in affected_nodes:
                    affected_nodes.add(current)
                    # Add nodes that depend on current node to stack
                    stack.extend([n for n in G_copy.predecessors(current) 
                                if n not in affected_nodes])

            # Measure impact
            if G_copy.number_of_nodes() > 0:
                largest_wcc = len(max(nx.weakly_connected_components(G_copy), key=len))
                relative_size = largest_wcc / initial_nodes
                components = nx.number_weakly_connected_components(G_copy)
                remaining_edges = G_copy.number_of_edges()
                edge_density = remaining_edges / initial_edges if initial_edges > 0 else 0
            else:
                largest_wcc = 0
                relative_size = 0
                components = 0
                remaining_edges = 0
                edge_density = 0

            # Store results with same metrics as node removal
            results.append({
                "edges_removed": i + 1,
                "nodes_affected": len(affected_nodes),
                "cascade_multiplier": len(affected_nodes) / (i + 1),
                "largest_component_size": largest_wcc,
                "relative_size": relative_size,
                "num_components": components,
                "remaining_edges": remaining_edges
            })

            if i % 100 == 0:
                print(f"Removed edge {i+1}, affected {len(affected_nodes)} nodes")

        print(f"Completed random edge removal simulation after removing {len(results)} edges")
        return results
    
    def analyze_pagerank_during_cascade(self, n_remove=2502):
        """
        Analyze how PageRank distribution changes during cascade failure, providing insight
        into network vulnerability and hierarchy.
        
        Parameters:
            n_remove (int): Number of nodes to remove (default matches previous analysis)
        
        Returns:
            Dictionary containing PageRank analysis metrics over cascade progression
        """
        print("Analyzing PageRank distribution during cascade...")
        G_copy = self.G.copy()
        initial_nodes = G_copy.number_of_nodes()
        results = []
        
        # Calculate initial PageRank distribution
        initial_pagerank = nx.pagerank(G_copy)
        initial_max = max(initial_pagerank.values())
        initial_mean = np.mean(list(initial_pagerank.values()))
        
        # Track nodes by their initial PageRank for removal
        sorted_nodes = sorted(initial_pagerank.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(0, n_remove, 100):  # Sample every 100 removals to keep computation manageable
            # Remove next batch of highest PageRank nodes
            for j in range(i, min(i + 100, n_remove)):
                if j < len(sorted_nodes):
                    node = sorted_nodes[j][0]
                    if node in G_copy:
                        # Find affected nodes through dependency chain
                        affected_nodes = set()
                        stack = [node]
                        while stack:
                            current = stack.pop()
                            if current not in affected_nodes:
                                affected_nodes.add(current)
                                stack.extend([n for n in G_copy.predecessors(current) 
                                        if n not in affected_nodes])
                        
                        G_copy.remove_nodes_from(affected_nodes)
            
            if G_copy.number_of_nodes() > 0:
                # Calculate new PageRank distribution
                current_pagerank = nx.pagerank(G_copy)
                pagerank_values = list(current_pagerank.values())
                
                # Calculate concentration metrics
                max_pagerank = max(pagerank_values)
                mean_pagerank = np.mean(pagerank_values)
                
                # Calculate inequality metrics (how concentrated influence is)
                sorted_values = sorted(pagerank_values)
                n = len(sorted_values)
                if n > 0:
                    gini = sum(i * val for i, val in enumerate(sorted_values))
                    gini = 2 * gini / (n * sum(sorted_values)) - (n + 1) / n
                else:
                    gini = 0
                    
                results.append({
                    "nodes_removed": i,
                    "nodes_remaining": G_copy.number_of_nodes(),
                    "max_pagerank": max_pagerank,
                    "mean_pagerank": mean_pagerank,
                    "pagerank_gini": gini,
                    "max_pagerank_ratio": max_pagerank / initial_max if initial_max > 0 else 0,
                    "mean_pagerank_ratio": mean_pagerank / initial_mean if initial_mean > 0 else 0
                })
        
        print("PageRank analysis completed.")
        return results

    def plot_pagerank_evolution(self, results):
        """
        Create visualization of PageRank metrics evolution during cascade failure.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot PageRank concentration metrics
        x = [r["nodes_removed"] for r in results]
        ax1.plot(x, [r["max_pagerank"] for r in results], 'b-', label='Max PageRank')
        ax1.plot(x, [r["mean_pagerank"] for r in results], 'r-', label='Mean PageRank')
        ax1.set_xlabel('Nodes Removed')
        ax1.set_ylabel('PageRank Value')
        ax1.set_title('PageRank Concentration During Cascade')
        ax1.grid(True)
        ax1.legend()
        
        # Plot inequality metrics
        ax2.plot(x, [r["pagerank_gini"] for r in results], 'g-', label='Gini Coefficient')
        ax2.plot(x, [r["max_pagerank_ratio"] for r in results], 'purple', 
                label='Max PageRank Ratio to Initial', linestyle='--')
        ax2.set_xlabel('Nodes Removed')
        ax2.set_ylabel('Inequality Metric')
        ax2.set_title('Network Inequality Evolution During Cascade')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    def simulate_incremental_weak_tie_removal(self, step_percentage=0.5):
        total_nodes = 2502
        step_size = int((step_percentage / 100) * total_nodes)
        sorted_nodes = sorted(self.G.degree, key=lambda x: x[1])[:total_nodes]  # Weak-tie nodes (low degree)
        
        G_copy = self.G.copy()
        results = []
        
        for i in range(0, len(sorted_nodes), step_size):
            # Remove 0.5% of weakest nodes
            nodes_to_remove = [n[0] for n in sorted_nodes[i:i + step_size]]
            G_copy.remove_nodes_from(nodes_to_remove)
            
            # Measure metrics
            largest_cc = len(max(nx.strongly_connected_components(G_copy), key=len)) if len(G_copy.nodes()) > 0 else 0
            num_components = nx.number_strongly_connected_components(G_copy)
            remaining_edges = G_copy.number_of_edges()
            
            # Track results
            results.append({
                "nodes_removed": len(nodes_to_remove) * (i // step_size + 1),
                "largest_component_size": largest_cc,
                "num_components": num_components,
                "remaining_edges": remaining_edges
            })
            
            print(f"Step {i // step_size + 1}: Removed {len(nodes_to_remove)} nodes")

        return results

    def plot_cascade_results(self, results, removal_type="nodes", strategy="rich-club"):
        """
        Plot the results of a cascade simulation.
        
        Parameters:
            results: List of dictionaries with simulation metrics.
            removal_type: "nodes" or "edges".
            strategy: Name of the failure strategy (e.g., "rich-club").
        """
        x = [r[f"{removal_type}_removed"] for r in results]
        largest_component = [r["largest_component_size"] for r in results]
        num_components = [r["num_components"] for r in results]
        remaining_edges = [r["remaining_edges"] for r in results]

        plt.figure(figsize=(10, 6))

        # Plot largest component size
        plt.plot(x, largest_component, label="Largest Component Size", marker='o')

        # Plot number of components
        plt.plot(x, num_components, label="Number of Components", marker='s')

        # Plot remaining edges
        plt.plot(x, remaining_edges, label="Remaining Edges", marker='^')

        plt.xlabel(f"{removal_type.capitalize()} Removed")
        plt.ylabel("Impact Metrics")
        plt.title(f"Cascade Simulation ({removal_type.capitalize()} Removal): {strategy}")
        plt.legend()
        plt.grid(True)
        # plt.show()





def main():
    # Create visualizer
    visualizer = NetworkVisualizer('pypi_nodes_20241216_081109.csv', 'pypi_edges_20241216_081109.csv')
    
    # Generate all plots
    # Print network statistics first
    visualizer.print_network_stats()
    
    # Try different k-core values
    # for k in [2, 3, 4, 5]:
    #     print(f"\nTrying k-core = {k}")
    #     plot = visualizer.plot_network_structure(k_core=k)
    #     if plot:
    #         plot.savefig(f'network_structure_k{k}.png', dpi=300, bbox_inches='tight')
    
    # rich_club_plot = visualizer.plot_rich_club()
    # rich_club_plot.savefig('rich_club.png', dpi=300, bbox_inches='tight')
    # Plot rich club analysis for in-degree
    # For the rich club analysis with normalization
    # rich_club_plot, rich_club_data = visualizer.plot_rich_club(degree_type="in", n_random=10)
    # rich_club_plot.savefig('rich_club_normalized_in.png', dpi=300, bbox_inches='tight')
    # print(f"Rich club data for in-degree: {rich_club_data}")

    # You might want to do the same for out-degree
    # rich_club_plot_out, rich_club_data_out = visualizer.plot_rich_club(degree_type="out", n_random=10)
    # print(f"Rich club data for out-degree: {rich_club_data_out}")
    # rich_club_plot_out.savefig('rich_club_normalized_out.png', dpi=300, bbox_inches='tight')

    # # To analyze the composition at different thresholds
    # thresholds = [10, 20, 30, 40]
    # for k in thresholds:
    #     metrics_in = visualizer.analyze_rich_club_composition(degree_type="in", k_threshold=k)
    #     print(f"Metrics for in-degree > {k}: {metrics_in}")
    #     metrics_out = visualizer.analyze_rich_club_composition(degree_type="out", k_threshold=k)
    #     print(f"Metrics for out-degree > {k}: {metrics_out}")

    # betweenness_plot = visualizer.plot_betweenness_vs_degree()
    # betweenness_plot.savefig('betweenness_degree.png', dpi=300, bbox_inches='tight')

    # degree_dist_plot = visualizer.plot_degree_distribution()
    # degree_dist_plot.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')
    # visualizer.analyze_assortativity()
    # visualizer.analyze_directed_assortativity()
    try:
        n_remove = 2502
        # Basic connectivity and component analysis
        # print(f"Number of strongly connected components: {nx.number_strongly_connected_components(visualizer.G)}")
        # print(f"Size of largest strongly connected component: {len(max(nx.strongly_connected_components(visualizer.G), key=len))}")

        # # Clustering
        # avg_clustering = nx.average_clustering(visualizer.G)
        # print(f"Average clustering coefficient: {avg_clustering}")

        # # For a large directed network like PyPI, sampling might be needed for path lengths
        # # Take the largest strongly connected component for path analysis
        # largest_cc = max(nx.strongly_connected_components(visualizer.G), key=len)
        # largest_cc_graph = visualizer.G.subgraph(largest_cc)

        # try:
        #     # These might be computationally expensive
        #     diameter = nx.diameter(largest_cc_graph)
        #     avg_path = nx.average_shortest_path_length(largest_cc_graph)
        #     print(f"Network diameter (largest SCC): {diameter}")
        #     print(f"Average shortest path length (largest SCC): {avg_path}")
        # except:
        #     print("Path calculations too computationally intensive - consider sampling")

        # # Assortativity (which you already have)
        # print("Degree Assortativity:")
        # print(f"in-in: {nx.degree_assortativity_coefficient(visualizer.G, x='in', y='in')}")
        # print(f"in-out: {nx.degree_assortativity_coefficient(visualizer.G, x='in', y='out')}")
        # print(f"out-in: {nx.degree_assortativity_coefficient(visualizer.G, x='out', y='in')}")
        # print(f"out-out: {nx.degree_assortativity_coefficient(visualizer.G, x='out', y='out')}")

        zero_in_degree = sum(1 for _, d in visualizer.G.in_degree() if d == 0)
        total_nodes = visualizer.G.number_of_nodes()
        percentage = (zero_in_degree / total_nodes) * 100

        print(f"Nodes with in-degree 0: {zero_in_degree}")
        print(f"Percentage of total nodes: {percentage:.2f}%")
        density = nx.density(visualizer.G)
        print(f"Full network density: {density}")
        # page_rank_results = visualizer.simulate_cascade_failure_weak(n_remove=n_remove, strategy="pagerank")

        # hubs_results = visualizer.simulate_cascade_failure_weak(n_remove=n_remove, strategy="hubs")

        #auth_results = visualizer.simulate_cascade_failure_weak(n_remove=n_remove, strategy="authorities")

        random_results = visualizer.simulate_cascade_failure_weak_node_degree(n_remove=n_remove, strategy="random")

        edge_removal_total = int((n_remove/582431) * 1781247)
        print(f"Total edges to remove: {edge_removal_total}")
        results = visualizer.simulate_random_edge_removal_node_degree(n_remove=edge_removal_total)
        # results_page_rank = visualizer.analyze_pagerank_during_cascade(n_remove=n_remove)
        # visualizer.plot_pagerank_evolution(results_page_rank)

    #     # targeted attack: rich nodes
    #     rich_club_nodes = visualizer.get_rich_club_nodes(degree_type="in", degree_threshold=50)
        
    # #     # Step 2: Simulate failures
    #     results = visualizer.simulate_rich_club_failure_batched(rich_club_nodes)

      #     #targeted attack bridge nodes
        # bridge_nodes = visualizer.get_bridge_nodes(max_in_degree=10)
        # bridge_results = visualizer.simulate_bridge_failure(bridge_nodes)

    finally:
        #     # Get the current date and time
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        visualizer.plot_cascade_results(results, removal_type="edges", strategy="random")
        plt.savefig(f'edge_removal_{current_time}.png')
        # plt.savefig(f'page_rank_stats{current_time}.png')
        
    #     # Step 3: Plot the results
        # visualizer.plot_cascade_results(results, removal_type="nodes", strategy="rich-club")
        # visualizer.plot_rich_club_characteristics(results)
        # plt.savefig(f'rich_club_stats{current_time}.png')

        # visualizer.plot_cascade_results(results)
        # plt.savefig(f'rich_club_cascade{current_time}.png')
    #     # targeted attack weak ties
    #     results = visualizer.simulate_incremental_weak_tie_removal(step_percentage=0.5)
    #     visualizer.plot_cascade_results(results, removal_type="nodes", strategy="weak-tie")
        # visualizer.plot_cascade_results(page_rank_results, removal_type="nodes", strategy="pagerank")
        # plt.savefig(f'pagerank_plot_{current_time}.png')

        # visualizer.plot_cascade_results(hubs_results, removal_type="nodes", strategy="hubs")
        # plt.savefig(f'hubs_plot_{current_time}.png')

        # visualizer.plot_cascade_results(auth_results, removal_type="nodes", strategy="authorities")
        # plt.savefig(f'authorities_plot_{current_time}.png')

        visualizer.plot_cascade_results(random_results, removal_type="nodes", strategy="random")
        plt.savefig(f'random_plot_{current_time}.png')


        # visualizer.plot_cascade_results(bridge_results, removal_type="nodes", strategy="bridge-nodes")
        # plt.savefig(f'bridge_nodes_plot_{current_time}.png')
        # bridge_nodes = visualizer.get_stratified_low_degree_nodes(max_in_degree=10)
        # bridge_results = visualizer.simulate_bridge_failure(bridge_nodes)

        # visualizer.plot_cascade_results(bridge_results, removal_type="nodes", strategy="bridge-nodes")
        # plt.savefig(f'bridge_nodes_plot_{current_time}.png')

    
    # Save plots
    # network_plot.savefig('network_structure.png', dpi=300, bbox_inches='tight')
    # rich_club_plot.savefig('rich_club.png', dpi=300, bbox_inches='tight')
    # betweenness_plot.savefig('betweenness_degree.png', dpi=300, bbox_inches='tight')
    # degree_dist_plot.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()









##############################################
# Degree distribution analysis
##############################################

def generate_synthetic_networks(original_graph, num_nodes):
    """
    Generate synthetic networks: Barabási–Albert (BA) and Erdős–Rényi (Random)
    """
    # Number of edges to attach for BA model
    logging.info("Generating synthetic networks...")
    logging.info(f"Type of input graph before: {type(original_graph)}")
    avg_degree = int(np.mean([deg for _, deg in original_graph.degree()]))
    logging.info("generating BA")
    ba_graph = nx.barabasi_albert_graph(num_nodes, avg_degree)
    ba_graph = nx.DiGraph(ba_graph)  # Convert to directed graph
    logging.info("generating ER")
    probability = (avg_degree / num_nodes)
    er_graph = nx.fast_gnp_random_graph(num_nodes, probability)
    er_graph = nx.DiGraph(er_graph)  # Convert to directed graph
    logging.info("Synthetic networks generated.")
    return ba_graph, er_graph

def generate_synthetic_networks_directed(original_graph, num_nodes):
    """
    Generate directed synthetic networks: Barabási–Albert (BA) and Erdős–Rényi (Random)
    """
    # Calculate average in-degree for parameters
    print("Generating synthetic networks...")
    avg_in_degree = int(np.mean([deg for _, deg in original_graph.in_degree()]))
    
    # Generate directed random graph directly
    probability = avg_in_degree / num_nodes
    print("Generating ER model...")
    er_graph = nx.fast_gnp_random_graph(n=num_nodes, p=probability, directed=True)
    
    # Generate directed BA graph
    print("Generating BA model...")
    ba_graph = nx.DiGraph()
    # Start with a small complete directed graph
    m = avg_in_degree
    ba_graph.add_nodes_from(range(m))
    for i in range(m):
        for j in range(m):
            if i != j:
                ba_graph.add_edge(i, j)
    
    # Add remaining nodes with preferential attachment
    for source in range(m, num_nodes):
        # Add new node
        ba_graph.add_node(source)
        # Choose targets based on in-degree
        targets = list(ba_graph.nodes())
        weights = [ba_graph.in_degree(target) + 1 for target in targets]
        probs = np.array(weights) / sum(weights)
        # Select m unique targets
        selected_targets = np.random.choice(
            targets, 
            size=min(m, len(targets)), 
            replace=False, 
            p=probs
        )
        # Add edges
        for target in selected_targets:
            ba_graph.add_edge(source, target)
    
    return ba_graph, er_graph

def generate_synthetic_networks_directed_bidirectional(original_graph, num_nodes):
    """
    Generate directed synthetic networks with both in and out degree preferential attachment.
    This creates a more realistic model where nodes can connect based on both in and out degrees.
    
    Parameters:
        original_graph: NetworkX graph to base parameters on
        num_nodes: Number of nodes for synthetic networks
        
    Returns:
        ba_graph: Directed Barabási-Albert graph with in/out degree attachment
        er_graph: Directed Erdős-Rényi graph
    """
    logging.info("Generating synthetic networks...")
    
    # Calculate average degrees
    avg_in_degree = int(np.mean([deg for _, deg in original_graph.in_degree()]))
    avg_out_degree = int(np.mean([deg for _, deg in original_graph.out_degree()]))
    m = max(avg_in_degree, avg_out_degree)
    
    # Generate ER graph
    logging.info("Generating ER model...")
    probability = avg_in_degree / num_nodes
    er_graph = nx.fast_gnp_random_graph(n=num_nodes, p=probability, directed=True)
    
    # Generate BA graph with both in and out degree preferential attachment
    logging.info("Generating BA model with in/out degree attachment...")
    ba_graph = nx.DiGraph()
    
    # Initialize with a complete graph of m nodes
    for i in range(m):
        for j in range(m):
            if i != j:
                ba_graph.add_edge(i, j)
    logging.info("Initialized with complete graph.")
    
    # Process nodes in batches
    batch_size = 1000
    for batch_start in range(m, num_nodes, batch_size):
        batch_end = min(batch_start + batch_size, num_nodes)
        batch_size_actual = batch_end - batch_start
        
        if batch_start % 1000 == 0:
            logging.info(f"Processing nodes {batch_start} to {batch_end}...")
        
        # Get current network state
        existing_nodes = np.array(list(ba_graph.nodes()))
        
        # Calculate both in and out degree probabilities
        in_degrees = np.array([ba_graph.in_degree(n) + 1 for n in existing_nodes])
        out_degrees = np.array([ba_graph.out_degree(n) + 1 for n in existing_nodes])
        
        # Combine in and out degree probabilities
        p_in = in_degrees / in_degrees.sum()
        p_out = out_degrees / out_degrees.sum()
        p_combined = (p_in + p_out) / 2  # Equal weight to in and out degrees
        
        # Generate edges for this batch
        new_edges = []
        for source in range(batch_start, batch_end):
            # Add edges based on combined probability
            targets = np.random.choice(
                existing_nodes,
                size=min(m, len(existing_nodes)),
                replace=False,
                p=p_combined
            )
            new_edges.extend((source, target) for target in targets)
            
            # Occasionally add reverse edges to create more realistic bidirectional connections
            if np.random.random() < 0.3:  # 30% chance of reverse edges
                reverse_targets = np.random.choice(
                    existing_nodes,
                    size=min(m//2, len(existing_nodes)),  # Fewer reverse edges
                    replace=False,
                    p=p_combined
                )
                new_edges.extend((target, source) for target in reverse_targets)
        
        # Add all nodes and edges from this batch
        ba_graph.add_nodes_from(range(batch_start, batch_end))
        ba_graph.add_edges_from(new_edges)
    
    logging.info("Synthetic networks generated.")
    return ba_graph, er_graph

# def plot_degree_comparison(original_graph, ba_graph, er_graph):
#     """
#     Plot degree distributions of the original graph, BA, and Random graphs
#     """
#     print("Plotting degree distributions...")
#     def degree_distribution(graph):
#         degrees = [d for _, d in graph.degree()]
#         unique_degrees, counts = np.unique(degrees, return_counts=True)
#         return unique_degrees, counts

#     def plot_power_law_fit(degrees, counts, ax, label):
#         # Remove zero degrees for log-log scaling
#         non_zero = degrees > 0
#         degrees, counts = degrees[non_zero], counts[non_zero]

#         # Fit power-law
#         def power_law(x, a, b):
#             return a * np.power(x, b)

#         try:
#             popt, _ = curve_fit(power_law, degrees, counts)
#             x_fit = np.linspace(min(degrees), max(degrees), 100)
#             y_fit = power_law(x_fit, *popt)
#             ax.plot(x_fit, y_fit, linestyle='--', label=f'{label} (fit: b={popt[1]:.2f})')
#         except:
#             ax.text(0.5, 0.9, f"Fit failed for {label}", transform=ax.transAxes, fontsize=8)

#     # Degree distributions
#     fig, ax = plt.subplots(figsize=(10, 6))

#     for graph, label in zip([original_graph, ba_graph, er_graph], 
#                             ['Original PyPI', 'BA Model', 'Random']):
#         degrees, counts = degree_distribution(graph)
#         ax.scatter(degrees, counts, label=label, s=10, alpha=0.7)
#         plot_power_law_fit(degrees, counts, ax, label)

#     # Formatting
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlabel('Degree (log)')
#     ax.set_ylabel('Count (log)')
#     ax.set_title('Degree Distribution Comparison')
#     ax.legend()
#     plt.grid(True)
#     plt.show()

# # Main Execution
# visualizer = NetworkVisualizer('final_csv/pypi_nodes_20241216_081109.csv', 'final_csv/pypi_edges_20241216_081109.csv')
# pypi_graph = visualizer.G
# num_nodes = pypi_graph.number_of_nodes()

# # Generate synthetic networks
# ba_graph, er_graph = generate_synthetic_networks(pypi_graph, num_nodes)

# # Compare degree distributions
# plot_degree_comparison(pypi_graph, ba_graph, er_graph)

def degree_distribution(graph, degree_type="in"):
    """
    Compute degree distribution for a given graph and degree type.
    
    Parameters:
    - graph: NetworkX graph
    - degree_type: "in" for in-degree, "out" for out-degree
    
    Returns:
    - unique_degrees: Array of unique degree values
    - counts: Array of counts corresponding to each degree
    """
    if degree_type == "in":
        degrees = [d for _, d in graph.in_degree() if d > 0]
    elif degree_type == "out":
        degrees = [d for _, d in graph.out_degree() if d > 0]
    else:
        raise ValueError("degree_type must be 'in' or 'out'")
    
    degrees = np.array(degrees)
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    return unique_degrees, counts

def plot_degree_distributions(G, ba_graph, er_graph):
    """
    Plot in-degree and out-degree distributions for PyPI and synthetic networks.
    """
    def power_law(x, a, b):
        """Power law function for curve fitting."""
        return a * np.power(x, b)

    for graph, name in zip([G, ba_graph, er_graph], ["Original PyPI", "BA Model", "ER Model"]):
        print(f"Analyzing {name}...")

        # Compute in-degrees and out-degrees
        in_degrees = [d for _, d in graph.in_degree() if d > 0]
        out_degrees = [d for _, d in graph.out_degree() if d > 0]

        # Plot distributions
        for degrees, degree_type in zip([in_degrees, out_degrees], ["In-Degree", "Out-Degree"]):
            unique_degrees, counts = np.unique(degrees, return_counts=True)

            # Debugging
            print(f"Unique Degrees: {unique_degrees}")
            print(f"Counts: {counts}")
            if len(unique_degrees) < 2:
                print(f"Insufficient data for power law fitting in {degree_type} ({name}).")
                continue

            # Ensure valid data
            if any(d <= 0 for d in unique_degrees) or any(c <= 0 for c in counts):
                print(f"Invalid values detected in degrees or counts for {degree_type} ({name}).")
                continue

            plt.figure(figsize=(8, 6))
            plt.scatter(unique_degrees, counts, alpha=0.7, label=f"{name} Data")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(f"{degree_type} (log)")
            plt.ylabel("Count (log)")
            plt.title(f"{degree_type} Distribution ({name})")

            # Fit power law
            try:
                popt, _ = curve_fit(power_law, unique_degrees, counts, maxfev=10000)
                x_fit = np.linspace(min(unique_degrees), max(unique_degrees), 100)
                y_fit = power_law(x_fit, *popt)
                plt.plot(x_fit, y_fit, "r--", label=f"Power law fit (b={popt[1]:.2f})")
            except Exception as e:
                print(f"Power law fitting failed for {degree_type} ({name}): {e}")
                plt.text(0.5, 0.9, "Power law fit failed", transform=plt.gca().transAxes, fontsize=8)
            
            plt.legend()
            plt.grid(True)
            plt.show()

def plot_degree_distributions_overlay(G, ba_graph, er_graph):
    """
    Plot in-degree and out-degree distributions overlayed for PyPI, BA, and ER graphs.
    """
    def power_law(x, a, b):
        """Power law function for curve fitting."""
        return a * np.power(x, b)

    for graph, name in zip([G, ba_graph, er_graph], ["Original PyPI", "BA Model", "ER Model"]):
        print(f"Analyzing {name}...")

        # Compute in-degrees and out-degrees
        in_degrees = [d for _, d in graph.in_degree() if d > 0]
        out_degrees = [d for _, d in graph.out_degree() if d > 0]

        # Unique degrees and counts for in/out-degree
        in_degrees_unique, in_counts = np.unique(in_degrees, return_counts=True)
        out_degrees_unique, out_counts = np.unique(out_degrees, return_counts=True)

        plt.figure(figsize=(10, 6))

        # Plot in-degree distribution
        plt.scatter(in_degrees_unique, in_counts, alpha=0.7, label="In-Degree", color="blue")
        plt.xscale("log")
        plt.yscale("log")

        # Plot out-degree distribution
        plt.scatter(out_degrees_unique, out_counts, alpha=0.7, label="Out-Degree", color="orange")
        
        # Fit power law for in-degree
        if len(in_degrees_unique) > 1:
            popt, _ = curve_fit(power_law, in_degrees_unique, in_counts, maxfev=10000)
            x_fit = np.linspace(min(in_degrees_unique), max(in_degrees_unique), 100)
            y_fit = power_law(x_fit, *popt)
            plt.plot(x_fit, y_fit, "b--", label=f"In-Degree Fit (b={popt[1]:.2f})")

        # Fit power law for out-degree
        if len(out_degrees_unique) > 1:
            popt, _ = curve_fit(power_law, out_degrees_unique, out_counts, maxfev=10000)
            x_fit = np.linspace(min(out_degrees_unique), max(out_degrees_unique), 100)
            y_fit = power_law(x_fit, *popt)
            plt.plot(x_fit, y_fit, "r--", label=f"Out-Degree Fit (b={popt[1]:.2f})")

        plt.xlabel("Degree (log)")
        plt.ylabel("Count (log)")
        plt.title(f"In-Degree vs Out-Degree Distribution ({name})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{name}_overlay.png", dpi=300)
        plt.show()

def plot_degree_distributions_side_by_side(G, ba_graph, er_graph):
    """
    Plot in-degree and out-degree distributions side by side for PyPI, BA, and ER graphs.
    """
    def power_law(x, a, b):
        """Power law function for curve fitting."""
        return a * np.power(x, b)

    for graph, name in zip([G, ba_graph, er_graph], ["Original PyPI", "BA Model", "ER Model"]):
        print(f"Analyzing {name}...")

        # Compute in-degrees and out-degrees
        in_degrees = [d for _, d in graph.in_degree() if d > 0]
        out_degrees = [d for _, d in graph.out_degree() if d > 0]

        # Unique degrees and counts for in/out-degree
        in_degrees_unique, in_counts = np.unique(in_degrees, return_counts=True)
        out_degrees_unique, out_counts = np.unique(out_degrees, return_counts=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # In-Degree Plot
        axes[0].scatter(in_degrees_unique, in_counts, alpha=0.7, label="In-Degree", color="blue")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("In-Degree (log)")
        axes[0].set_ylabel("Count (log)")
        axes[0].set_title(f"In-Degree Distribution ({name})")
        axes[0].grid(True)

        # Fit power law for in-degree
        if len(in_degrees_unique) > 1:
            popt, _ = curve_fit(power_law, in_degrees_unique, in_counts, maxfev=10000)
            x_fit = np.linspace(min(in_degrees_unique), max(in_degrees_unique), 100)
            y_fit = power_law(x_fit, *popt)
            axes[0].plot(x_fit, y_fit, "b--", label=f"Fit (b={popt[1]:.2f})")
        axes[0].legend()

        # Out-Degree Plot
        axes[1].scatter(out_degrees_unique, out_counts, alpha=0.7, label="Out-Degree", color="orange")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Out-Degree (log)")
        axes[1].set_ylabel("Count (log)")
        axes[1].set_title(f"Out-Degree Distribution ({name})")
        axes[1].grid(True)

        # Fit power law for out-degree
        if len(out_degrees_unique) > 1:
            popt, _ = curve_fit(power_law, out_degrees_unique, out_counts, maxfev=10000)
            x_fit = np.linspace(min(out_degrees_unique), max(out_degrees_unique), 100)
            y_fit = power_law(x_fit, *popt)
            axes[1].plot(x_fit, y_fit, "r--", label=f"Fit (b={popt[1]:.2f})")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"{name}_side_by_side.png", dpi=300)
        plt.show()

def plot_separate_in_out_degrees(G, ba_graph, er_graph):
    """
    Plot separate in-degree and out-degree distributions for Original PyPI, BA, and Random graphs.
    """
    def power_law(x, a, b):
        """Power law function for curve fitting."""
        return a * np.power(x, b)

    # Names for graphs
    graph_names = ["Original PyPI", "BA Model", "Random"]
    graphs = [G, ba_graph, er_graph]

    # In-Degree Plot
    plt.figure(figsize=(10, 6))
    for graph, name in zip(graphs, graph_names):
        in_degrees = [d for _, d in graph.in_degree() if d > 0]
        unique_degrees, counts = np.unique(in_degrees, return_counts=True)

        plt.scatter(unique_degrees, counts, alpha=0.7, label=f"{name} In-Degree")
        plt.xscale("log")
        plt.yscale("log")
        
        # Fit power law
        if len(unique_degrees) > 1:
            try:
                popt, _ = curve_fit(power_law, unique_degrees, counts, maxfev=10000)
                x_fit = np.linspace(min(unique_degrees), max(unique_degrees), 100)
                y_fit = power_law(x_fit, *popt)
                plt.plot(x_fit, y_fit, linestyle="--", label=f"{name} Fit (b={popt[1]:.2f})")
            except Exception as e:
                print(f"Power law fitting failed for {name} In-Degree: {e}")

    plt.xlabel("In-Degree (log)")
    plt.ylabel("Count (log)")
    plt.title("In-Degree Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("In_Degree_Distribution.png", dpi=300)
    plt.show()

    # Out-Degree Plot
    plt.figure(figsize=(10, 6))
    for graph, name in zip(graphs, graph_names):
        out_degrees = [d for _, d in graph.out_degree() if d > 0]
        unique_degrees, counts = np.unique(out_degrees, return_counts=True)

        plt.scatter(unique_degrees, counts, alpha=0.7, label=f"{name} Out-Degree")
        plt.xscale("log")
        plt.yscale("log")
        
        # Fit power law
        if len(unique_degrees) > 1:
            try:
                popt, _ = curve_fit(power_law, unique_degrees, counts, maxfev=10000)
                x_fit = np.linspace(min(unique_degrees), max(unique_degrees), 100)
                y_fit = power_law(x_fit, *popt)
                plt.plot(x_fit, y_fit, linestyle="--", label=f"{name} Fit (b={popt[1]:.2f})")
            except Exception as e:
                print(f"Power law fitting failed for {name} Out-Degree: {e}")

    plt.xlabel("Out-Degree (log)")
    plt.ylabel("Count (log)")
    plt.title("Out-Degree Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Out_Degree_Distribution.png", dpi=300)
    # plt.show()


# Main Execution
# visualizer = NetworkVisualizer('pypi_nodes_20241216_081109.csv', 'pypi_edges_20241216_081109.csv')
# pypi_graph = visualizer.G
# # Check a sample of edges
# logging.info("Sample edges (directed):", list(pypi_graph.edges(data=True))[:5])
# # Try accessing in_degree directly

# num_nodes = pypi_graph.number_of_nodes()

# # # Generate synthetic networks
# ba_graph, er_graph = generate_synthetic_networks(pypi_graph, num_nodes)
# ba_graph, er_graph = generate_synthetic_networks_directed_bidirectional(pypi_graph, num_nodes)

# # # Plot in-degree and out-degree distributions
# # try:
# #     plot_degree_distributions(pypi_graph, ba_graph, er_graph)
# # except Exception as e:
# #     print(f"Error during degree distribution plotting: {e}")
# plot_separate_in_out_degrees(pypi_graph, ba_graph, er_graph)
