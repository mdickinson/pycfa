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

from typing import Dict, Set

from pycfa.cfgraph import CFGraph
from pycfa.cfnode import CFNode

#: Labels for particular identified nodes.
ENTERC = "enterc"
LEAVE = "leave"
RAISEC = "raisec"
RETURN = "return"
RETURN_VALUE = "return_value"

# Type alias for analysis contexts.
Context = Dict[str, CFNode]


class CFAnalysis:
    """
    The control-flow analysis produced by the analyzer.
    """

    #: The control-flow graph.
    _graph: CFGraph[CFNode]

    #: Index to important nodes: mapping from labels to particular
    #: nodes.
    context: Context

    def __init__(self, graph: CFGraph[CFNode], context: Context) -> None:
        self._graph = graph
        self.context = context

    # Graph inspection methods.

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

    @property
    def entry_node(self) -> CFNode:
        """
        Return the entry node for this analysis.
        """
        return self.context[ENTERC]

    @property
    def raise_node(self) -> CFNode:
        """
        Return the raise node for this analysis.

        Raises
        ------
        AttributeError
            If there's no raise node, because the function cannot raise.
        """
        if RAISEC in self.context:
            return self.context[RAISEC]
        else:
            raise AttributeError("No raise node")

    @property
    def leave_node(self) -> CFNode:
        """
        Return the leave node for this analysis (relevant only for modules
        and classes).
        """
        if LEAVE in self.context:
            return self.context[LEAVE]
        else:
            raise AttributeError("No leave node")

    @property
    def return_with_value(self) -> CFNode:
        """
        The return node for paths that return with a value.

        Raises
        ------
        AttributeError
            If no path returns with a value.
        """
        if RETURN_VALUE in self.context:
            return self.context[RETURN_VALUE]
        else:
            raise AttributeError("No returns with a value")

    @property
    def return_without_value(self) -> CFNode:
        """
        The return node for paths that return with a value.

        Raises
        ------
        AttributeError
            If no path returns with a value.
        """
        if RETURN in self.context:
            return self.context[RETURN]
        else:
            raise AttributeError("No returns without a value")
