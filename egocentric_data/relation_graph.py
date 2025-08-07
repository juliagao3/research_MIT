import random
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import pandas as pd


class RelationGraph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.adjacency_list = {node: set() for node in nodes}
        self.incoming_count = {node: 0 for node in nodes}
        self.root = None

    def generate_random_edges(self):
        shuffled = self.nodes[:]
        random.shuffle(shuffled)

        self.root = shuffled.pop(0)

        available = shuffled[:]
        random.shuffle(available)
        num_branches = min(random.randint(3, 4), len(available))
        frontier = []

        for _ in range(num_branches):
            child = available.pop(0)
            self.adjacency_list[self.root].add(child)
            self.incoming_count[child] += 1
            frontier.append(child)

        while available and frontier:
            parent = frontier.pop(0)
            if len(available) >= 3:  
                num_children = 2
            elif len(available) >= 1:
                num_children = 1
            else:
                break

            for _ in range(num_children):
                if not available:
                    break
                child = available.pop(0)
                self.adjacency_list[parent].add(child)
                self.adjacency_list[child].add(parent)
                self.incoming_count[child] += 1
                frontier.append(child)

    def get_root(self):
        return self.root
    
    def egocentric_traversal(self, start_node, visited=None, depth=0):
        if visited is None:
            visited = set()

        indent = "  " * depth
        neighbors = sorted(self.adjacency_list[start_node] - visited)
        print(f"{indent}- {start_node} (neighbors: {', '.join(neighbors)})")
        visited.add(start_node)

        for neighbor in neighbors:
            self.egocentric_traversal(neighbor, visited, depth + 1)


def save_full_relation_graph(graph, output_path="full_walk_views/full_relation_graph.png"):
    parent = os.path.dirname(output_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    G = nx.Graph()
    for node, neighbors in graph.adjacency_list.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    try:
        pos = graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=10,
        edge_color="gray"
    )
    plt.title("Full Relation Graph")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def full_exploratory_walk(graph, walk_idx, output_dir="full_walk_views", stop_prob=0.2):
    walk_dir = os.path.join(output_dir, f"walk_{walk_idx}")
    os.makedirs(walk_dir, exist_ok=True)

    G = nx.Graph()
    for node, neighbors in graph.adjacency_list.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    visited = set()
    walk = []

    # Start at a random node
    current = random.choice(graph.nodes)
    stack = [current]

    step = 0
    while stack:
        current = stack[-1]
        walk.append(current)

        # Save egocentric view
        sub_nodes = [current] + list(graph.adjacency_list[current])
        H = G.subgraph(sub_nodes)

        try:
            pos = graphviz_layout(H, prog="dot")
        except:
            pos = nx.spring_layout(H, seed=42)

        plt.figure(figsize=(6, 4))
        node_colors = ["orange" if n == current else "lightblue" for n in H.nodes]
        nx.draw(
            H, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=2000,
            font_size=10
        )
        plt.title(f"Step {step}: {current}")
        fname = f"{walk_dir}/walk_step_{step}_{current.replace(' ', '_')}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

        visited.add(current)
        step += 1

        # Check for unvisited neighbors
        neighbors = [n for n in graph.adjacency_list[current] if n not in visited]

        if len(walk) >= 7 and neighbors and random.random() < stop_prob:
            break
        
        if neighbors:
            stack.append(random.choice(neighbors))
        else:
            stack.pop()  # backtrack

    with open(f"{walk_dir}/trajectory.txt", "w") as f:
        f.write("\n".join(walk))

    df = pd.DataFrame(
        [[nx.shortest_path_length(G, walk[i], walk[j]) 
          for j in range(len(walk))] 
         for i in range(len(walk))],
        index=walk, columns=walk
    )
    df.to_csv(os.path.join(walk_dir, "distance_matrix.csv"))


    return walk


topics = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]
 
graph = RelationGraph(topics)
graph.generate_random_edges()
print("Root node:", graph.get_root())
graph.egocentric_traversal(graph.get_root())

save_full_relation_graph(graph)

for i in range(5):
    walk = full_exploratory_walk(graph, walk_idx=i)
    print("Visited walk:", " → ".join(walk))