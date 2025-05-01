import pandas as pd
import networkx as nx

# Load the CSV file
df = pd.read_csv('MaxFlow_Graph.csv')

# Create a directed graph
G = nx.DiGraph()

# Add edges from the DataFrame
for index, row in df.iterrows():
    G.add_edge(row['source'], row['target'], capacity=row['capacity'])

# Compute the maximum flow from source 'S' to sink 'Z' (replace with your actual nodes)
flow_value, flow_dict = nx.maximum_flow(G, 'S', 'Z')

print(f"Maximum flow: {flow_value}")
print("Flow paths:")
for u in flow_dict:
    for v in flow_dict[u]:
        if flow_dict[u][v] > 0:
            print(f"{u} -> {v}: {flow_dict[u][v]}")
