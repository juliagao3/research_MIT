import random
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import pandas as pd
import string


class RelationGraph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.adjacency_list = {node: set() for node in nodes}
        self.incoming_count = {node: 0 for node in nodes}
        self.root = None

    def generate_random_edges(self, min_branches=2, max_branches=4):
        shuffled = self.nodes[:]
        random.shuffle(shuffled)

        self.root = shuffled.pop(0)
        available = shuffled[:]
        random.shuffle(available)

        num_branches = 0
        if available:
            num_branches = min(random.randint(min_branches, max_branches), len(available))
        frontier = []

        #connect all nodes
        for _ in range(num_branches):
            child = available.pop(0)
            self.adjacency_list[self.root].add(child)
            self.incoming_count[child] += 1
            frontier.append(child)

        while available:
            if not frontier:
                # if exhausted (e.g., very small initial branching), pick any existing node as parent
                frontier.append(random.choice(list(self.adjacency_list.keys())))
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

def full_exploratory_walk(graph, walk_idx, output_dir="easy_walk", stop_prob=0.2, min_steps=7, max_steps=None):
    walk_dir = os.path.join(output_dir, f"walk_{walk_idx}")
    os.makedirs(walk_dir, exist_ok=True)

    #build graph
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

    if max_steps is None:
        max_steps = max(min_steps + 3, int(0.8 * len(graph.nodes)))


    while stack and len(walk) < max_steps:
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

        if len(walk) >= min_steps and neighbors:
            if random.random() < stop_prob:
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


def make_node_labels(n):
    labels = []
    alphabet = list(string.ascii_uppercase)
    k = 0
    while len(labels) < n:
        if k < 26:
            labels.append(alphabet[k])
        else:
            first = (k // 26) - 1
            second = k % 26
            labels.append(alphabet[first] + alphabet[second])
        k += 1
    return labels[:n]


# for j in range(20):
#     graph = RelationGraph(topics)
#     graph.generate_random_edges()
#     print("Root node:", graph.get_root())
#     graph.egocentric_traversal(graph.get_root())

#     for i in range(5):
#         walk = full_exploratory_walk(graph, f"easy_walk{j}", walk_idx=i)
#         print("Visited walk:", " → ".join(walk))

#     save_full_relation_graph(graph)
def generate_dataset(out_root="relation_graph_dataset", easy_range=(15, 20), hard_range=(30, 36), 
                     walks_per_graph=5, easy_step=(10, 15), hard_step=(25, 30), easy_stop=0.3, hard_stop=0.1):
    
    os.makedirs(out_root, exist_ok=True)

    #easy graphs
    easy_root = os.path.join(out_root, "easy")
    os.makedirs(easy_root, exist_ok=True)

    for j in range(20):
        n_nodes = random.randint(*easy_range)
        topics = make_node_labels(n_nodes)
        graph = RelationGraph(topics)
        graph.generate_random_edges()

        print(f"[EASY {j}] Root node:", graph.get_root())
        graph.egocentric_traversal(graph.get_root())

        graph_dir = os.path.join(easy_root, f"graph_{j}")
        os.makedirs(graph_dir, exist_ok=True)

        save_full_relation_graph(
            graph,
            output_path=os.path.join(graph_dir, "full_relation_graph.png")
        )

        for i in range(walks_per_graph):
            min_steps = random.randint(*easy_step)
            walk = full_exploratory_walk(
                graph,
                walk_idx=i,
                output_dir=graph_dir,
                stop_prob=easy_stop,
                min_steps=min_steps,
                max_steps=max(min_steps + 3, n_nodes)  # cap by graph size
            )
            print(f"[EASY {j}] Walk {i}: {' → '.join(walk)}")

    # hard_root = os.path.join(out_root, "hard")
    # os.makedirs(hard_root, exist_ok=True)


    # for j in range(10):
    #     n_nodes = random.randint(*hard_range)
    #     topics = make_node_labels(n_nodes)
    #     graph = RelationGraph(topics)
    #     graph.generate_random_edges()

    #     print(f"[HARD {j}] Root node:", graph.get_root())
    #     graph.egocentric_traversal(graph.get_root())

    #     graph_dir = os.path.join(hard_root, f"graph_{j}")
    #     os.makedirs(graph_dir, exist_ok=True)

    #     save_full_relation_graph(
    #         graph,
    #         output_path=os.path.join(graph_dir, "full_relation_graph.png")
    #     )

    #     for i in range(walks_per_graph):
    #         min_steps = random.randint(*hard_step)
    #         walk = full_exploratory_walk(
    #             graph,
    #             walk_idx=i,
    #             output_dir=graph_dir,
    #             stop_prob=hard_stop,
    #             min_steps=min_steps,
    #             max_steps=max(min_steps + 6, int(0.8 * n_nodes))  # cap by graph size
    #         )
    #         print(f"[HARD {j}] Walk {i}: {' → '.join(walk)}")

if __name__ == "__main__":
    generate_dataset(out_root="relation_graph_dataset", easy_range=(15, 20), hard_range=(30, 36), 
                     walks_per_graph=5, easy_step=(10, 15), hard_step=(25, 30), easy_stop=0.3, hard_stop=0.1)