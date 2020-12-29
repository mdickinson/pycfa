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
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""
from __future__ import annotations

import ast
import contextlib
from typing import Dict, List, Mapping, Optional, Union

from pycfa.cfanalysis import CFAnalysis, ENTERC, LEAVE, RAISEC, RETURN, RETURN_VALUE
from pycfa.cfgraph import CFGraph
from pycfa.cfnode import CFNode

# Context labels
BREAK = "break"
CONTINUE = "continue"

# Edge labels

# There are four possible labels, and four node types. The set of edge labels
# for a given node depends only on the node type.
#
# - GENERIC nodes have outward labels NEXT and RAISE. Most non-control-flow
#   statements correspond to generic nodes.
# - SIMPLE nodes have only a NEXT label (e.g., break, continue, pass, return, try)
# - RAISE nodes have only RAISE labels
# - BRANCH nodes have outward labels ENTER, ELSE and RAISE. These are used in
#   if, for, while and try/except statements.

#: Link to the next statement (if no errors occurred)
NEXT = "next"

#: Link followed if an error is raised.
RAISE = "raise"

#: Link followed to enter an if / for / while / except block
ENTER = "enter"

#: Link followed when a condition does not apply
ELSE = "else"


# Type alias for analysis contexts.
Context = Dict[str, CFNode]


class CFAnalyser:
    """
    The control-flow analysis.

    This is a directed graph (not necessarily acyclic) with labelled edges.
    Most nodes will correspond directly to an AST statement.
    """

    #: The control-flow graph.
    _graph: CFGraph[CFNode]

    #: Current context, while graph is under construction.
    _context: Context

    def __init__(self) -> None:
        self._graph = CFGraph()
        self._context = {}

    # Graph building methods.

    def _new_node(
        self,
        *,
        edges: Optional[Dict[str, CFNode]] = None,
        ast_node: Optional[ast.AST] = None,
        annotation: Optional[str] = None,
    ) -> CFNode:
        """
        Create a new control-flow node and add it to the graph.

        Parameters
        ----------
        edges : dict
            Mapping from edge labels to target nodes.
        ast_node : ast.AST, optional
            Linked ast node.
        annotation : str, optional
            Text annotation for the node; used for nodes that aren't
            linked to an AST node.

        Returns
        -------
        node : CFNode
            The newly-created node.
        """
        if edges is None:
            edges = {}

        node = CFNode(ast_node=ast_node, annotation=annotation)
        self._graph.add_node(node, edges=edges)
        return node

    @contextlib.contextmanager
    def _updated_context(self, updates: Mapping):
        """
        Temporarily update the context dictionary with the given values.
        """
        context = self._context
        original_items = {
            label: context[label] for label in updates if label in context
        }
        try:
            self._context.update(updates)
            yield
        finally:
            for label in updates:
                if label in original_items:
                    context[label] = original_items.pop(label)
                else:
                    del context[label]

    # General analysis helpers.

    def _analyse_simple(self, statement: ast.stmt, *, next: CFNode) -> CFNode:
        """
        Analyse a simple non-control-flow statement that can't raise.

        Examples are pass, global, nonlocal.
        """
        return self._new_node(edges={NEXT: next}, ast_node=statement)

    def _analyse_generic(self, statement: ast.stmt, *, next: CFNode) -> CFNode:
        """
        Analyse a generic statement that doesn't affect control flow.
        """
        return self._new_node(
            edges={RAISE: self._context[RAISEC], NEXT: next},
            ast_node=statement,
        )

    def _analyse_loop(
        self,
        statement: Union[ast.AsyncFor, ast.For, ast.While],
        *,
        next: CFNode,
    ) -> CFNode:
        """
        Analyse a loop statement (for, async for, or while).
        """
        # Node acting as target for the next iteration. We'll identify this
        # with the loop entry node, once that exists.
        dummy_node = self._new_node()
        with self._updated_context({BREAK: next, CONTINUE: dummy_node}):
            body_node = self._analyse_statements(statement.body, next=dummy_node)

        loop_node = self._new_node(
            edges={
                RAISE: self._context[RAISEC],
                ELSE: self._analyse_statements(statement.orelse, next=next),
                ENTER: body_node,
            },
            ast_node=statement,
        )

        self._graph.collapse_node(dummy_node, loop_node)
        return loop_node

    def _analyse_statements(self, statements: List[ast.stmt], *, next) -> CFNode:
        """
        Analyse a sequence of statements.
        """
        for statement in reversed(statements):
            analyse = getattr(self, "analyse_" + type(statement).__name__)
            next = analyse(statement, next=next)
        return next

    def _analyse_try_except_else(self, statement: ast.Try, *, next: CFNode) -> CFNode:
        """
        Analyse the try-except-else part of a try statement. The finally
        part is ignored, as though it weren't present.
        """
        # Process handlers backwards; raise if the last handler doesn't match.
        raise_node = self._context[RAISEC]
        for handler in reversed(statement.handlers):
            match_node = self._analyse_statements(handler.body, next=next)
            if handler.type is None:
                # Bare except always matches, never raises.
                raise_node = match_node
            else:
                raise_node = self._new_node(
                    edges={
                        RAISE: self._context[RAISEC],
                        ENTER: match_node,
                        ELSE: raise_node,
                    },
                    ast_node=handler.type,
                )

        else_node = self._analyse_statements(statement.orelse, next=next)

        with self._updated_context({RAISEC: raise_node}):
            body_node = self._analyse_statements(statement.body, next=else_node)

        return self._new_node(edges={NEXT: body_node}, ast_node=statement)

    def _analyse_with(
        self,
        statement: Union[ast.AsyncWith, ast.With],
        *,
        next: CFNode,
    ) -> CFNode:
        """
        Analyse a with or async with statement.
        """
        with_node = self._new_node(
            edges={
                ENTER: self._analyse_statements(
                    statement.body,
                    next=next,
                ),
                RAISE: self._context[RAISEC],
            },
            ast_node=statement,
        )
        return with_node

    # Statement analyzers for particular AST node types.

    def analyse_AnnAssign(self, statement: ast.AnnAssign, *, next: CFNode) -> CFNode:
        """
        Analyse an annotated assignment statement.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_Assert(self, statement: ast.Assert, *, next: CFNode) -> CFNode:
        """
        Analyse an assert statement.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_Assign(self, statement: ast.Assign, *, next: CFNode) -> CFNode:
        """
        Analyse an assignment statement.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_AsyncFor(self, statement: ast.AsyncFor, *, next: CFNode) -> CFNode:
        """
        Analyse an async for statement.
        """
        return self._analyse_loop(statement, next=next)

    def analyse_AsyncFunctionDef(
        self, statement: ast.AsyncFunctionDef, *, next: CFNode
    ) -> CFNode:
        """
        Analyse an async function (coroutine) definition.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_AsyncWith(self, statement: ast.AsyncWith, *, next: CFNode) -> CFNode:
        """
        Analyse an async with statement.
        """
        return self._analyse_with(statement, next=next)

    def analyse_AugAssign(self, statement: ast.AugAssign, *, next: CFNode) -> CFNode:
        """
        Analyse an augmented assignment statement.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_Break(self, statement: ast.Break, *, next: CFNode) -> CFNode:
        """
        Analyse a break statement.
        """
        return self._new_node(edges={NEXT: self._context[BREAK]}, ast_node=statement)

    def analyse_ClassDef(self, statement: ast.ClassDef, *, next: CFNode) -> CFNode:
        """
        Analyse a class definition.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_Continue(self, statement: ast.Continue, *, next: CFNode) -> CFNode:
        """
        Analyse a continue statement.
        """
        return self._new_node(edges={NEXT: self._context[CONTINUE]}, ast_node=statement)

    def analyse_Delete(self, statement: ast.Delete, *, next: CFNode) -> CFNode:
        """
        Analyse a del statement.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_Expr(self, statement: ast.Expr, *, next: CFNode) -> CFNode:
        """
        Analyse an expression (used as a statement).
        """
        return self._analyse_generic(statement, next=next)

    def analyse_For(self, statement: ast.For, *, next: CFNode) -> CFNode:
        """
        Analyse a for statement.
        """
        return self._analyse_loop(statement, next=next)

    def analyse_FunctionDef(
        self, statement: ast.FunctionDef, *, next: CFNode
    ) -> CFNode:
        """
        Analyse a function definition.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_Global(self, statement: ast.Global, *, next: CFNode) -> CFNode:
        """
        Analyse a global statement
        """
        return self._analyse_simple(statement, next=next)

    def analyse_If(self, statement: ast.If, *, next: CFNode) -> CFNode:
        """
        Analyse an if statement.
        """
        return self._new_node(
            edges={
                ENTER: self._analyse_statements(statement.body, next=next),
                ELSE: self._analyse_statements(statement.orelse, next=next),
                RAISE: self._context[RAISEC],
            },
            ast_node=statement,
        )

    def analyse_Import(self, statement: ast.Import, *, next: CFNode) -> CFNode:
        """
        Analyse an import statement.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_ImportFrom(self, statement: ast.ImportFrom, *, next: CFNode) -> CFNode:
        """
        Analyse a from ... import statement.
        """
        return self._analyse_generic(statement, next=next)

    def analyse_Nonlocal(self, statement: ast.Nonlocal, *, next: CFNode) -> CFNode:
        """
        Analyse a nonlocal statement.
        """
        return self._analyse_simple(statement, next=next)

    def analyse_Pass(self, statement: ast.Pass, *, next: CFNode) -> CFNode:
        """
        Analyse a pass statement.
        """
        return self._analyse_simple(statement, next=next)

    def analyse_Raise(self, statement: ast.Raise, *, next: CFNode) -> CFNode:
        """
        Analyse a raise statement.
        """
        return self._new_node(edges={RAISE: self._context[RAISEC]}, ast_node=statement)

    def analyse_Return(self, statement: ast.Return, *, next: CFNode) -> CFNode:
        """
        Analyse a return statement.
        """
        if statement.value is None:
            return self._new_node(
                edges={NEXT: self._context[RETURN]}, ast_node=statement
            )
        else:
            return self._new_node(
                edges={NEXT: self._context[RETURN_VALUE], RAISE: self._context[RAISEC]},
                ast_node=statement,
            )

    def analyse_Try(self, statement: ast.Try, *, next: CFNode) -> CFNode:
        """
        Analyse a try statement.
        """
        # To analyse a try-except-else-finally block, it's easier to think
        # of it as two separate pieces: it's equivalent to the try-except-else
        # piece, nested *inside* a try-finally.

        # The finally block can be entered by various different means (from
        # a return, raise, break or continue in the try-except-else part, or
        # from completing the try-except-else without error). For each of
        # these different entrances to the finally block, the target on
        # *leaving* the finally block may be different. So we will
        # construct different paths in the control flow graph for each of
        # the possible leave targets for the finally block. (This is similar
        # to the way that Python >= 3.9 bytecode handles the finally.)

        # We also want to avoid generating *all* six possible finally paths:
        # some of them may not be relevant. So we adopt the following approach:

        # 1. Create dummy target nodes for the try-except-else, and analyse
        #    the try-except-else with those dummy nodes.
        # 2. For only the dummy nodes that are actually reached, construct
        #    the corresponding finally paths.
        # 3. Link up the graphs.

        # For each actual node in the context (excluding duplicates),
        # create a corresponding dummy node.
        dummy_nodes = {node: self._new_node() for node in set(self._context.values())}
        dummy_nodes[next] = self._new_node()

        # Analyse the try-except-else part of the statement using those dummy
        # nodes in place of the real ones.
        try_except_else_context = {
            key: dummy_nodes[node] for key, node in self._context.items()
        }
        with self._updated_context(try_except_else_context):
            entry_node = self._analyse_try_except_else(
                statement, next=dummy_nodes[next]
            )

        # Now iterate through the dummy nodes. For those that aren't reached,
        # remove them. For those that are, replace with the corresponding
        # finally code. Note that there will always be at least one of the
        # dummy nodes reachable from the try-except-else, so we'll always
        # analyse the finally code at least once.

        for end_node, dummy_node in dummy_nodes.items():
            if self._graph.edges_to(dummy_node):
                # Dummy node is reachable from the try-except-else.
                # Create a corresponding finally block.
                finally_node = self._analyse_statements(
                    statement.finalbody, next=end_node
                )
                self._graph.collapse_node(dummy_node, finally_node)
            else:
                # Dummy node not reachable; remove it.
                self._graph.remove_node(dummy_node)

        return entry_node

    def analyse_While(self, statement: ast.While, *, next: CFNode) -> CFNode:
        """
        Analyse a while statement
        """
        return self._analyse_loop(statement, next=next)

    def analyse_With(self, statement: ast.With, *, next: CFNode) -> CFNode:
        """
        Analyse a with statement.
        """
        return self._analyse_with(statement, next=next)

    def analyse_class(self, ast_node: ast.ClassDef) -> CFAnalysis:
        """
        Construct a control flow graph for an ast.Module node.
        """
        leave_node = self._new_node(annotation="<leave>")
        raise_node = self._new_node(annotation="<raise>")

        with self._updated_context({RAISEC: raise_node}):
            enter_node = self._analyse_statements(ast_node.body, next=leave_node)

        self._new_node(edges={ENTER: enter_node}, annotation="<start>")

        context = {ENTERC: enter_node}
        if self._graph._backedges[raise_node]:
            context[RAISEC] = raise_node
        if self._graph._backedges[leave_node]:
            context[LEAVE] = leave_node

        return CFAnalysis(graph=self._graph, context=context)

    def analyse_function(
        self, ast_node: Union[ast.AsyncFunctionDef, ast.FunctionDef]
    ) -> CFAnalysis:
        """
        Construct a control flow graph for a function or coroutine AST node.

        Parameters
        ----------
        ast_node : ast.FunctionDef or ast.AsyncFunctionDef
            Function node in the ast tree of the code being analysed.
        """
        return_node = self._new_node(annotation="<return-without-value>")
        return_value_node = self._new_node(annotation="<return-with-value>")
        raise_node = self._new_node(annotation="<raise>")

        with self._updated_context(
            {RAISEC: raise_node, RETURN: return_node, RETURN_VALUE: return_value_node}
        ):
            enter_node = self._analyse_statements(ast_node.body, next=return_node)

        # Make sure there's at least one reference to the first node.
        self._new_node(edges={ENTER: enter_node}, annotation="<start>")

        context = {ENTERC: enter_node}
        if self._graph._backedges[raise_node]:
            context[RAISEC] = raise_node
        if self._graph._backedges[return_value_node]:
            context[RETURN_VALUE] = return_value_node
        if self._graph._backedges[return_node]:
            context[RETURN] = return_node

        return CFAnalysis(graph=self._graph, context=context)

    def analyse_module(self, ast_node: ast.Module) -> CFAnalysis:
        """
        Construct a control flow graph for an ast.Module node.
        """
        leave_node = self._new_node(annotation="<leave>")
        raise_node = self._new_node(annotation="<raise>")

        with self._updated_context({RAISEC: raise_node}):
            enter_node = self._analyse_statements(ast_node.body, next=leave_node)

        self._new_node(edges={ENTER: enter_node}, annotation="<start>")

        context = {ENTERC: enter_node}
        if self._graph._backedges[leave_node]:
            context[LEAVE] = leave_node
        if self._graph._backedges[raise_node]:
            context[RAISEC] = raise_node

        return CFAnalysis(graph=self._graph, context=context)
