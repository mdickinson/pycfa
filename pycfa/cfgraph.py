"""
Graph structure for the control-flow graph.
"""

import ast
from typing import Dict, Optional, Set, Tuple


class CFNode:
    """
    A node on the control flow graph.
    """

    ast_node: Optional[ast.AST]

    def __init__(self, ast_node: Optional[ast.AST] = None) -> None:
        self.ast_node = ast_node


class CFGraph:
    """
    The directed graph underlying the control flow graph.
    """

    _nodes: Set[CFNode]
    _edges: Dict[CFNode, Dict[str, CFNode]]
    _backedges: Dict[CFNode, Set[Tuple[CFNode, str]]]

    def __init__(self) -> None:
        self._nodes = set()
        self._edges = {}
        self._backedges = {}

    def add_node(self, node: CFNode) -> None:
        """
        Add a node to the graph. Raises on an attempt to add a node that
        already exists.
        """
        assert node not in self._nodes
        self._nodes.add(node)
        self._edges[node] = {}
        self._backedges[node] = set()

    def remove_node(self, node: CFNode) -> None:
        """
        Remove a node from the graph. Fails if there are edges to or
        from that node.
        """
        assert not self._backedges[node]
        assert not self._edges[node]
        self._nodes.remove(node)

    def add_edge(self, source: CFNode, label: str, target: CFNode) -> None:
        """
        Add a labelled edge to the graph. Raises if an edge from the given
        source, with the given label, already exists.
        """
        assert label not in self._edges[source]
        self._edges[source][label] = target

        assert (source, label) not in self._backedges[target]
        self._backedges[target].add((source, label))

    def remove_edge(self, source: CFNode, label: str, target: CFNode) -> None:
        self._backedges[target].remove((source, label))
        self._edges[source].pop(label)

    def edge(self, source: CFNode, label: str) -> CFNode:
        """
        Get the target of a given edge.
        """
        return self._edges[source][label]

    def edge_labels(self, source: CFNode) -> Set[str]:
        """
        Get labels of all edges.
        """
        return set(self._edges[source].keys())

    def edges_to(self, target: CFNode) -> Set[Tuple[CFNode, str]]:
        """
        Set of pairs (source, label) representing edges to this node.
        """
        return self._backedges[target]
