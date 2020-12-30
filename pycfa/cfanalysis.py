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
Result of a control flow analysis applied to a module,
function, coroutine or class.
"""

from typing import Iterable, Optional, Set

from pycfa.cfgraph import CFGraph
from pycfa.cfnode import CFNode


class CFAnalysis:
    """
    The control-flow analysis produced by the analyzer.
    """

    #: The first node of the module, function or class.
    entry_node: CFNode

    #: Dummy node representing the exit point of a module, function or class.
    #: For functions, this node is reached on a plain "return" or on the implicit
    #: return at the end of the function body. Functions that return a value
    #: will reach return_node instead.
    leave_node: Optional[CFNode]

    #: Dummy node representing an uncaught exception in a module, function or class.
    #: This attribute may be missing, but will not be None.
    raise_node: Optional[CFNode]

    #: Dummy node representing an explicit return-with-value from a function.
    return_node: Optional[CFNode]

    #: The control-flow graph.
    _graph: CFGraph[CFNode]

    def __init__(
        self,
        graph: CFGraph[CFNode],
        *,
        entry_node: CFNode,
        raise_node: Optional[CFNode] = None,
        leave_node: Optional[CFNode] = None,
        return_node: Optional[CFNode] = None,
    ) -> None:
        self._graph = graph
        self.entry_node = entry_node
        self.raise_node = raise_node
        self.leave_node = leave_node
        self.return_node = return_node

    # Graph inspection methods.

    def nodes(self) -> Iterable[CFNode]:
        """
        Iterable for the collection of all nodes.
        """
        return self._graph._nodes

    def edge(self, source: CFNode, label: str) -> CFNode:
        """
        Get the target of a given edge.
        """
        return self._graph.edge(source, label)

    def edge_labels(self, source: CFNode) -> Set[str]:
        """
        Get labels of all edges.
        """
        return self._graph.edge_labels(source)
