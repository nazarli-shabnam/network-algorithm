import pandas as pd
import networkx as nx

df = pd.read_csv('MaxFlow_Graph.csv')

G = nx.DiGraph()

for index, row in df.iterrows():
    G.add_edge(row['Origin'], row['Destiny'], capacity=row['capacity'])

#max flow
flow_value, flow_dict = nx.maximum_flow(G, 'S', 'Z')

print(f"Maximum flow: {flow_value}")
print("Flow paths:")
for u in flow_dict:
    for v in flow_dict[u]:
        if flow_dict[u][v] > 0:
            print(f"{u} -> {v}: {flow_dict[u][v]}")
