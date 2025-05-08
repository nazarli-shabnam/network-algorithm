import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import time

try:
    cities_df = pd.read_csv('cities_in_az.csv')
    airports_df = pd.read_csv('airports.csv')
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure the files are in your Google Drive and update the paths.")

def plot_graph(G, title="Graph Visualization", figsize=(10, 8)):
    """Visualize the graph with matplotlib"""
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10)
    plt.title(title)
    plt.show()

def find_any_path(G, source, target):
    """Find any path from source to target using DFS"""
    visited = set()
    path = []

    def dfs(node):
        nonlocal path, visited
        if node == target:
            path.append(node)
            return True
        if node in visited:
            return False
        visited.add(node)
        path.append(node)
        for neighbor in G.neighbors(node):
            if dfs(neighbor):
                return True
        path.pop()
        return False

    if dfs(source):
        return path
    return None

def calculate_path_hours(G, path):
    """Calculate total hours for a given path in cities graph"""
    total_hours = 0
    for i in range(len(path)-1):
        total_hours += G.edges[path[i], path[i+1]].get('Hours', 0)
    return total_hours

def calculate_path_distance(G, path):
    """Calculate total distance for a given path in airports graph"""
    total_distance = 0
    for i in range(len(path)-1):
        total_distance += G.edges[path[i], path[i+1]].get('Distance', 0)
    return total_distance

def calculate_path_airtime(G, path):
    """Calculate total airtime for a given path in airports graph"""
    total_airtime = 0
    for i in range(len(path)-1):
        total_airtime += G.edges[path[i], path[i+1]].get('AirTime', 0)
    return total_airtime

def get_path_from_predecessors(predecessors, source, target):
    """Reconstruct path from predecessors dictionary"""
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = predecessors.get(node, None)
        if node == source:
            path.append(source)
            break
    return path[::-1] if path[0] == source else None

def bellman_ford(G, source, weight_attr=None):
    """Bellman-Ford algorithm implementation"""
    distances = {node: float('infinity') for node in G.nodes()}
    distances[source] = 0
    predecessors = {node: None for node in G.nodes()}

    for _ in range(len(G.nodes()) - 1):
        for u, v, data in G.edges(data=True):
            weight = data.get(weight_attr, 1) if weight_attr else 1
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u

    for u, v, data in G.edges(data=True):
        weight = data.get(weight_attr, 1) if weight_attr else 1
        if distances[u] + weight < distances[v]:
            raise ValueError("Graph contains a negative weight cycle")

    return distances, predecessors

def dijkstra(G, source, weight_attr=None):
    """Dijkstra's algorithm implementation"""
    distances = {node: float('infinity') for node in G.nodes()}
    distances[source] = 0
    predecessors = {node: None for node in G.nodes()}
    visited = set()

    queue = [(0, source)]

    while queue:
        current_dist, current_node = queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor in G.neighbors(current_node):
            edge_data = G.edges[current_node, neighbor]
            weight = edge_data.get(weight_attr, 1) if weight_attr else 1
            distance = current_dist + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                queue.append((distance, neighbor))

        queue.sort()

    return distances, predecessors

def cities_activities():
    print("\n=== Cities Graph Activities ===")

    G_cities = nx.from_pandas_edgelist(cities_df, source='Origin', target='Destiny', edge_attr=True)

    plot_graph(G_cities, "Cities Graph Visualization")

    source = 'Baku'
    target = 'Imishli'
    path = find_any_path(G_cities, source, target)
    print(f"\n3.1.2 Any path from {source} to {target}: {path}")
    if path:
        total_hours = calculate_path_hours(G_cities, path)
        print(f"3.1.3 Total hours for path: {total_hours}")

    shortest_path_unweighted = nx.shortest_path(G_cities, source=source, target=target)
    print(f"\n5.1.2 Shortest path (unweighted) from {source} to {target}: {shortest_path_unweighted}")

    shortest_path_weighted = nx.shortest_path(G_cities, source=source, target=target, weight='Hours')
    print(f"5.1.3 Shortest path (weighted by Hours) from {source} to {target}: {shortest_path_weighted}")

    if shortest_path_unweighted != shortest_path_weighted:
        print("\nThe weighted and unweighted paths are different because:")
        print("- Unweighted finds the path with fewest edges (hops)")
        print("- Weighted by 'Hours' finds the path with least total travel time")
    nx.add_path(G_cities, [source, target])
    G_cities.edges[source, target]['Hours'] = 1.29

    path_baku_to_imishli = nx.shortest_path(G_cities, source=source, target=target, weight='Hours')
    path_imishli_to_baku = nx.shortest_path(G_cities, source=target, target=source, weight='Hours')

    print(f"\n5.1.4 Path from {source} to {target}: {path_baku_to_imishli}")
    print(f"5.1.4 Path from {target} to {source}: {path_imishli_to_baku}")

    if path_baku_to_imishli != path_imishli_to_baku[::-1]:
        print("\nThe paths are different in each direction because:")
        print("- The graph is undirected (default in NetworkX unless specified otherwise)")
        print("- But edge weights might make different paths optimal in each direction")

def airports_activities():
    print("\n=== Airports Graph Activities ===")

    G_airports = nx.from_pandas_edgelist(airports_df, source='Origin', target='Dest',
                                        edge_attr=True, create_using=nx.DiGraph())

    print("\nPlotting a subgraph of airports (full graph might be too large)...")
    nodes_subset = list(G_airports.nodes())[:20]
    G_sub = G_airports.subgraph(nodes_subset)
    plot_graph(G_sub, "Airports Subgraph Visualization")

    source = 'CRP'
    target = 'BOI'

    if source not in G_airports.nodes() or target not in G_airports.nodes():
        print(f"\nWarning: {source} or {target} not found in airports graph. Using different nodes.")
        available_nodes = list(G_airports.nodes())
        source = available_nodes[0]
        target = available_nodes[-1]
        print(f"Using {source} to {target} instead.")

    path = find_any_path(G_airports, source, target)
    print(f"\n3.2.2 Any path from {source} to {target}: {path}")

    if path:
        total_distance = calculate_path_distance(G_airports, path)
        print(f"3.2.3 Total distance for path: {total_distance} miles")

    if path:
        total_airtime = calculate_path_airtime(G_airports, path)
        print(f"3.2.4 Total airtime for path: {total_airtime} minutes")

    try:
        shortest_path_distance = nx.shortest_path(G_airports, source=source, target=target, weight='Distance')
        dist = nx.shortest_path_length(G_airports, source=source, target=target, weight='Distance')
        print(f"\n5.2.2 Shortest path by distance from {source} to {target}: {shortest_path_distance}")
        print(f"Total distance: {dist} miles")
    except nx.NetworkXNoPath:
        print(f"\n5.2.2 No path exists from {source} to {target} by distance")

    try:
        shortest_path_airtime = nx.shortest_path(G_airports, source=source, target=target, weight='AirTime')
        airtime = nx.shortest_path_length(G_airports, source=source, target=target, weight='AirTime')
        print(f"\n5.2.3 Shortest path by airtime from {source} to {target}: {shortest_path_airtime}")
        print(f"Total airtime: {airtime} minutes")
    except nx.NetworkXNoPath:
        print(f"\n5.2.3 No path exists from {source} to {target} by airtime")

    print("\nNetwork Metrics Analysis:")

    print(f"\nDegree of {source}: {G_airports.degree(source)}")
    print(f"In-degree of {source}: {G_airports.in_degree(source)}")
    print(f"Out-degree of {source}: {G_airports.out_degree(source)}")

    closeness = nx.closeness_centrality(G_airports)
    print(f"\nCloseness centrality of {source}: {closeness.get(source, 0):.4f}")

    print("\nCalculating betweenness centrality (this might take a while)...")
    betweenness = nx.betweenness_centrality(G_airports, k=10)
    print(f"Betweenness centrality of {source}: {betweenness.get(source, 0):.4f}")

    density = nx.density(G_airports)
    print(f"\nNetwork density: {density:.4f}")

    print("\nCalculating diameter (this might take a while)...")
    try:
        diameter = nx.diameter(G_airports)
        print(f"Network diameter: {diameter}")
    except nx.NetworkXError:
        print("Graph is not connected - cannot compute diameter")

    print("\nCalculating average path length (this might take a while)...")
    try:
        avg_path_length = nx.average_shortest_path_length(G_airports)
        print(f"Average path length: {avg_path_length:.2f}")
    except nx.NetworkXError:
        print("Graph is not connected - cannot compute average path length")
    print("\n7.1 Comparing Bellman-Ford and Dijkstra's algorithms:")
    start_time = time.time()
    bf_distances, bf_predecessors = bellman_ford(G_airports, source, 'Distance')
    bf_path = get_path_from_predecessors(bf_predecessors, source, target)
    bf_time = time.time() - start_time

    start_time = time.time()
    dk_distances, dk_predecessors = dijkstra(G_airports, source, 'Distance')
    dk_path = get_path_from_predecessors(dk_predecessors, source, target)
    dk_time = time.time() - start_time

    print(f"\nDistance from {source} to {target}:")
    print(f"Bellman-Ford path: {bf_path}, distance: {bf_distances.get(target, 'inf')}, time: {bf_time:.4f}s")
    print(f"Dijkstra's path: {dk_path}, distance: {dk_distances.get(target, 'inf')}, time: {dk_time:.4f}s")

    start_time = time.time()
    bf_distances, bf_predecessors = bellman_ford(G_airports, source, 'AirTime')
    bf_path = get_path_from_predecessors(bf_predecessors, source, target)
    bf_time = time.time() - start_time
    start_time = time.time()
    dk_distances, dk_predecessors = dijkstra(G_airports, source, 'AirTime')
    dk_path = get_path_from_predecessors(dk_predecessors, source, target)
    dk_time = time.time() - start_time

    print(f"\nAirtime from {source} to {target}:")
    print(f"Bellman-Ford path: {bf_path}, airtime: {bf_distances.get(target, 'inf')}, time: {bf_time:.4f}s")
    print(f"Dijkstra's path: {dk_path}, airtime: {dk_distances.get(target, 'inf')}, time: {dk_time:.4f}s")

    print("\nPerformance comparison:")
    print("- Dijkstra's algorithm is generally faster for graphs with non-negative weights")
    print("- Bellman-Ford can handle negative weights but is slower")
    print("- For this airports graph with non-negative weights, Dijkstra's should be preferred")

if 'cities_df' in globals() and 'airports_df' in globals():
    cities_activities()
    airports_activities()
else:
    print("Could not run activities due to missing data files.")