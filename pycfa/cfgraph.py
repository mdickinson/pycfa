"""
Graph structure for the control-flow graph.

Conceptually, our graph is very similar to a DFA graph for a regular
expression. It consists of:

- a set of nodes
- for each node, a set of edge labels
- for each node and edge label, a target node

Note that it's possible to have parallel edges, and it's possible to
have self loops. The set of edge labels for a given node depends only
on the node type.

Each node can optionally contain a link to an underlying AST node. (But
note that distinct CFNodes can link to the same AST node.)

There are four possible labels, and four node types:

- GENERIC nodes have outward labels NEXT and RAISE
- SIMPLE nodes have only a NEXT label (
    e.g., break, continue, pass, return, try)
- RAISE nodes have only RAISE labels
- BRANCH nodes have outward labels ENTER, ELSE and RAISE

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

    # Functions that change the state of the graph.

    def new_node(
        self, edges: Dict[str, CFNode], ast_node: Optional[ast.AST] = None
    ) -> CFNode:
        """
        Create a new control-flow node and add it to the graph.

        Parameters
        ----------
        edges : dict
            Mapping from edge labels to target nodes.
        ast_node : ast.AST, optional
            Linked ast node.

        Returns
        -------
        node : CFNode
            The newly-created node.
        """
        node = CFNode(ast_node=ast_node)
        self._add_node(node)
        for name, target in edges.items():
            self._add_edge(node, name, target)
        return node

    def collapse_node(self, dummy: CFNode, target: CFNode) -> None:
        """
        Identify two nodes.

        Identifies the *dummy* node with the *target* node, and
        removes the *dummy* node from the graph. The dummy node
        should not have any outward edges.
        """
        assert not self._edges[dummy]

        edges_to_dummy = self.edges_to(dummy)
        for source, label in edges_to_dummy.copy():
            self._remove_edge(source, label)
            self._add_edge(source, label, target)

        self.remove_node(dummy)

    def remove_node(self, node: CFNode) -> None:
        """
        Remove a node from the graph. Fails if there are edges to or
        from that node.
        """
        assert not self._backedges[node]
        assert not self._edges[node]
        self._nodes.remove(node)

    # Functions for examining or traversing the graph.

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

    # Low-level functions

    def _add_node(self, node: CFNode) -> None:
        """
        Add a node to the graph. Raises on an attempt to add a node that
        already exists.
        """
        assert node not in self._nodes
        self._nodes.add(node)
        self._edges[node] = {}
        self._backedges[node] = set()

    def _add_edge(self, source: CFNode, label: str, target: CFNode) -> None:
        """
        Add a labelled edge to the graph. Raises if an edge from the given
        source, with the given label, already exists.
        """
        assert source in self._nodes
        assert target in self._nodes
        assert label not in self._edges[source]
        self._edges[source][label] = target

        assert (source, label) not in self._backedges[target]
        self._backedges[target].add((source, label))

    def _remove_edge(self, source: CFNode, label: str) -> None:
        """
        Remove a labelled edge from the graph.
        """
        target = self._edges[source][label]
        self._backedges[target].remove((source, label))
        self._edges[source].pop(label)
