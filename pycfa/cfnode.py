"""
The CFNode class provides the nodes to be used in the control-flow graph.

Each node optionally has a reference to an ast node, and optionally has a
textual annotation.
"""

import ast
from typing import Optional


class CFNode:
    """
    A node on the control flow graph.

    Parameters
    ----------
    ast_node
        Linked AST node
    annotation
        Text annotation
    """

    ast_node: Optional[ast.AST]

    annotation: Optional[str]

    def __init__(
        self,
        ast_node: Optional[ast.AST] = None,
        annotation: Optional[str] = None,
    ) -> None:
        self.ast_node = ast_node
        self.annotation = annotation
