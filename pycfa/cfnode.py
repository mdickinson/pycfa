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
