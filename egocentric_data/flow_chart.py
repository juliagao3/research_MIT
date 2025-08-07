import random
import operator
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne 
      }


class FlowNode:
    def __init__(self, name, question, condition, is_terminal=False):
        self.name = name
        self.question = question
        self.condition = condition
        self.edges = {}
        self.is_terminal = is_terminal

class FlowChart:
    def __init__(self, var_names):
        self.nodes = {}
        self.start = None
        self.var_names = var_names
        self.node_count = 0

    def __new_node(self, question, condition, is_terminal=False):
        name = f"Node{self.node_count}"
        self.node_count += 1
        node = FlowNode(name, question, condition, is_terminal)
        self.nodes[name] = node
        if self.start is None:
            self.start = node
        return node
    
    def generate(self, depth=5):
        start_node = self.__new_node(
                question="Start",
                condition=lambda _: True
            )

        input_vars = ', '.join(self.var_names.keys())
        input_node = self.__new_node(
            question=f"Input: {input_vars}",
            condition=lambda _: True
        )

        def _generate_recursive(depth):
            if depth == 0:
                return self.__new_node(
                    question="[end]",
                    condition=lambda _: True,
                    is_terminal=True
                )
            
            var1, var2 = random.sample(list(self.var_names.keys()), 2)
            op_str = random.choice(list(ops.keys()))
            condition = lambda context, v1=var1, v2=var2, op=ops[op_str]: op(context[v1], context[v2])
            question = f"{var1} {op_str} {var2}"

            node = self.__new_node(question=question, condition=condition)
            node.edges["true"] = _generate_recursive(depth - 1)
            node.edges["false"] = _generate_recursive(depth - 1)
            return node
        
        logic_root = _generate_recursive(depth)
        start_node.edges["next"] = input_node
        input_node.edges["next"] = logic_root
        self.root = start_node
    
    def traverse(self, input_values):
        path = []
        current = self.root
        while not current.is_terminal:
            result = current.condition(input_values)

            if current.question not in ["Start"] and not current.question.startswith("Input"):
                path.append((current.name, current.question, result))
            else:
                path.append((current.name, current.question, None))

            if "next" in current.edges:
                current = current.edges["next"]
            else:
                current = current.edges["true"] if result else current.edges["false"]
        path.append((current.name, current.question, None))
        return path
    
    def render(flowchart, highlight_path=None):
        G = nx.DiGraph()
        labels = {}

        # Build graph from flowchart nodes and edges
        for node in flowchart.nodes.values():
            labels[node.name] = node.question
            for label, child in node.edges.items():
                G.add_edge(node.name, child.name, label=label)

        # Layout: left to right
        pos = graphviz_layout(G, prog="dot")
        edge_labels = nx.get_edge_attributes(G, "label")

        # Node colors: lightblue for regular, lightgray for terminal
        node_colors = [
            "lightgray" if flowchart.nodes[node].is_terminal else "lightblue"
            for node in G.nodes()
        ]

        # Draw graph
        plt.figure(figsize=(16, 5))
        nx.draw(G, pos, with_labels=False, arrows=True, node_color=node_colors, node_size=2500)
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Flowchart (Left to Right)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


variables = {"a": 2, "b": 3, "c": 4}
chart = FlowChart(variables)
chart.generate()
traversal_path = chart.traverse(variables)

for name, question, result in traversal_path:
    if result is not None:
        print(f"{name}: {question} → {result}")
    else:
        print(f"{name}: {question}")

chart.render(chart)