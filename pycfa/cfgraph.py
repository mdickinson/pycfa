# Copyright 2020 Mark Dickinson. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The CFGraph class provides the graph structure for the control-flow graph.

Conceptually, our graph is very similar to a DFA graph for a regular
expression. It consists of:

- a set of nodes
- for each node, a set of edge labels (strings)
- for each node and edge label, a target node

The set of operations that can mutate the graph is very limited:

- a new node can be added, together with edges to existing nodes
- an isolated node can be removed
- a node with no outgoing edges can be identified with another node

Parallel edges (with different labels) and self-loops are permitted.
Nodes can be any hashable object.
"""

from typing import Dict, Generic, Mapping, Optional, Set, Tuple, TypeVar

#: Type of nodes. For now, require only that nodes are hashable.
NodeType = TypeVar("NodeType")


class CFGraph(Generic[NodeType]):
    """
    The directed graph underlying the control flow graph.
    """

    #: The collection of nodes.
    _nodes: Set[NodeType]

    #: Mapping from source node and edge label to target node.
    _edges: Dict[NodeType, Dict[str, NodeType]]

    #: Mapping from target node to collection of (source node, edge) pairs.
    _backedges: Dict[NodeType, Set[Tuple[NodeType, str]]]

    def __init__(self) -> None:
        self._nodes = set()
        self._edges = {}
        self._backedges = {}

    # Functions that change the state of the graph.

    def add_node(
        self,
        node: NodeType,
        *,
        edges: Optional[Mapping[str, NodeType]] = None,
    ):
        """
        Add a new node, along with edges to existing nodes to the graph.

        Parameters
        ----------
        node
            The node to be added to the graph.
        edges
            Edges from the given node, if any, provided as a mapping
            from edge labels (strings) to target nodes. The target nodes
            should already be in the graph.

        Raises
        ------
        ValueError
            If the given node is already in the graph, or if any of the target
            nodes for edges are not already in the graph.
        """
        if node in self:
            raise ValueError(f"node {node} is already present in the graph")

        self._add_node(node)
        if edges is not None:
            for label, target in edges.items():
                if target not in self or target == node:
                    raise ValueError(
                        f"target {target} for edge {label} is not in the graph"
                    )
                self._add_edge(node, label, target)

    def remove_node(self, node: NodeType) -> None:
        """
        Remove an isolated node from the graph.

        Fails if there are edges to or from that node: all edges must be removed
        before it's possible to remove the node itself.

        Parameters
        ----------
        node: NodeType
            The node to be removed.

        Raises
        ------
        ValueError
            If the node is not in the graph, or if the node is not isolated.
        """
        if node not in self._nodes:
            raise ValueError(f"node {node} not present in the graph")
        if self._edges[node]:
            raise ValueError(f"node {node} is not isolated: it has forward edges")
        if self._backedges[node]:
            raise ValueError(f"node {node} is not isolated: it has back edges")

        self._nodes.remove(node)

    def collapse_node(self, dummy: NodeType, target: NodeType) -> None:
        """
        Identify two nodes.

        Identifies the *dummy* node with the *target* node, and removes the
        *dummy* node from the graph. The dummy node should not have any outward
        edges.

        Note that this is the only mechanism for introducing cycles into the graph.

        Parameters
        ----------
        dummy
            Node to be collapsed and removed
        target
            Node to identify *dummy* with

        Raises
        ------
        ValueError
            If *dummy* has any outward edges, or if either of dummy or target is not
            in the graph.
        """
        if dummy not in self:
            raise ValueError(f"node {dummy} is not in the graph")

        if target not in self:
            raise ValueError(f"node {target} is not in the graph")

        if self._edges[dummy]:
            raise ValueError(f"node {dummy} has outward edges")

        if dummy == target:
            raise ValueError(f"nodes {dummy} and {target} must be distinct")

        edges_to_dummy = self.edges_to(dummy)
        for source, label in edges_to_dummy.copy():
            self._remove_edge(source, label)
            self._add_edge(source, label, target)

        self.remove_node(dummy)

    # Functions for examining or traversing the graph.

    def edge(self, source: NodeType, label: str) -> NodeType:
        """
        Get the target of a given edge.
        """
        return self._edges[source][label]

    def edge_labels(self, source: NodeType) -> Set[str]:
        """
        Get labels of all edges.
        """
        return set(self._edges[source].keys())

    def edges_to(self, target: NodeType) -> Set[Tuple[NodeType, str]]:
        """
        Set of pairs (source, label) representing edges to this node.
        """
        return self._backedges[target]

    # Support for membership testing

    def __contains__(self, node: NodeType) -> bool:
        """
        Determine whether a given node is contained in the graph.
        """
        return node in self._nodes

    # Low-level functions

    def _add_node(self, node: NodeType) -> None:
        """
        Add a node to the graph. Raises ValueError on an attempt to add a node that's
        already in the graph.
        """
        assert node not in self._nodes
        self._nodes.add(node)

        self._edges[node] = {}
        self._backedges[node] = set()

    def _add_edge(self, source: NodeType, label: str, target: NodeType) -> None:
        """
        Add a labelled edge to the graph.
        """
        assert label not in self._edges[source]
        self._edges[source][label] = target

        assert (source, label) not in self._backedges[target]
        self._backedges[target].add((source, label))

    def _remove_edge(self, source: NodeType, label: str) -> None:
        """
        Remove a labelled edge from the graph.
        """
        target = self._edges[source][label]
        self._backedges[target].remove((source, label))
        self._edges[source].pop(label)
