import pandas as pd
import networkx as nx
import time
import tracemalloc

df = pd.read_csv("MaxFlow_Graph.csv")

G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['Origin'], row['Destiny'], capacity=row['Capacity']) 

source = 'S' 
sink = 'Z'   

print("\n=== Edmonds-Karp Algorithm ===")
tracemalloc.start()
start_time = time.time()

flow_value, flow_dict = nx.maximum_flow(G, source, sink, flow_func=nx.algorithms.flow.edmonds_karp)

end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Max Flow: {flow_value}")
print(f"Time: {end_time - start_time:.6f} seconds")
print(f"Peak Memory: {peak / 1024:.2f} KB")
print("Augmented Paths:")
for u in flow_dict:
    for v in flow_dict[u]:
        if flow_dict[u][v] > 0:
            print(f"{u} -> {v}: {flow_dict[u][v]}")
