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
"""

import ast
import contextlib
from typing import Any, Dict, Generator, List, Tuple, Union

from pycfa.cfanalysis import CFAnalysis
from pycfa.cfgraph import CFGraph
from pycfa.cfnode import CFNode

# Edge labels

#: Link to the next statement (if no errors occurred)
NEXT = "next"

#: Link followed if an error is raised.
ERROR = "error"

#: Link followed to enter the body of an if / for / while / except / with block
ENTER = "enter"

#: Link followed when a condition does not apply
ELSE = "else_"

# Context labels, for internal use only.
_BREAK = "break_"
_CONTINUE = "continue_"
_RAISE = "raise_"
_LEAVE = "leave"
_RETURN = "return_"

# Type alias for analysis contexts.
_Context = Dict[str, CFNode]


class CFAnalyser:
    """
    Control-flow analyser.

    Analyses the AST node for a function, coroutine, module or class,
    returning a CFAnalysis.
    """

    #: The control-flow graph.
    _graph: CFGraph[CFNode]

    #: Current context, while graph is under construction.
    _context: _Context

    def __init__(self) -> None:
        self._graph = CFGraph()
        self._context = {}

    # Graph building methods.

    def _annotated_node(self, annotation: str, **edges: CFNode) -> CFNode:
        """
        Create a new control-flow node and add it to the graph.

        Parameters
        ----------
        annotation : str, optional
            Text annotation for the node.
        **edges : dict
            Mapping from edge labels to target nodes.

        Returns
        -------
        node : CFNode
            The newly-created node.
        """
        node = CFNode(annotation=annotation)
        self._graph.add_node(node, edges=edges)
        return node

    def _ast_node(self, statement: ast.AST, **edges: CFNode) -> CFNode:
        """
        Create a new node wrapping an AST node, with given edges to existing nodes.
        """
        node = CFNode(ast_node=statement)
        self._graph.add_node(node, edges=edges)
        return node

    def _dummy_node(self) -> CFNode:
        """
        Create a new dummy node, which will eventually be removed.
        """
        node = CFNode()
        self._graph.add_node(node)
        return node

    # Context management

    @contextlib.contextmanager
    def _updated_context(self, **updates: CFNode) -> Generator[None, None, None]:
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

    @property
    def _raise(self) -> CFNode:
        return self._context[_RAISE]

    # General analysis helpers.

    def _has_parents(self, node: CFNode) -> bool:
        """
        Determine whether the given node has direct parents.
        """
        return bool(self._graph._backedges[node])

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
        dummy_node = self._dummy_node()
        with self._updated_context(break_=next, continue_=dummy_node):
            body_node = self._analyse_statements(statement.body, next=dummy_node)

        loop_node = self._ast_node(
            statement,
            enter=body_node,
            else_=self._analyse_statements(statement.orelse, next=next),
            error=self._raise,
        )

        self._graph.collapse_node(dummy_node, loop_node)
        return loop_node

    def _analyse_statements(
        self, statements: List[ast.stmt], *, next: CFNode
    ) -> CFNode:
        """
        Analyse a sequence of statements.
        """
        for statement in reversed(statements):
            analyse = getattr(self, "_analyse_stmt_" + type(statement).__name__)
            next = analyse(statement, next=next)
        return next

    def _analyse_try_except(self, statement: ast.Try, *, next: CFNode) -> CFNode:
        """
        Analyse the try-except-else part of a try statement. The finally
        part is ignored, as though it weren't present.
        """
        # Process handlers backwards; raise if the last handler doesn't match.
        raise_node = self._raise
        for handler in reversed(statement.handlers):
            match_node = self._analyse_statements(handler.body, next=next)
            if handler.type is None:
                # Bare except always matches, never raises.
                raise_node = match_node
            else:
                raise_node = self._ast_node(
                    handler.type,
                    enter=match_node,
                    else_=raise_node,
                    error=self._raise,
                )

        else_node = self._analyse_statements(statement.orelse, next=next)

        with self._updated_context(raise_=raise_node):
            body_node = self._analyse_statements(statement.body, next=else_node)

        return self._ast_node(statement, next=body_node)

    def _analyse_with(
        self,
        statement: Union[ast.AsyncWith, ast.With],
        *,
        next: CFNode,
    ) -> CFNode:
        """
        Analyse a with or async with statement.
        """
        with_node = self._ast_node(
            statement,
            enter=self._analyse_statements(statement.body, next=next),
            error=self._raise,
        )
        return with_node

    def _expression_as_constant(self, expr: ast.expr) -> Tuple[bool, Any]:
        """
        Attempt to interpret the given expression as a compile-time constant.

        Returns a pair (is_constant, value) indicating whether the given expression
        could be interpreted as a constant or not, and its value if so. If is_constant
        is False, value is None.
        """
        method_name = "_getvalue_expr_" + type(expr).__name__
        if hasattr(self, method_name):
            getvalue = getattr(self, method_name)
            return True, getvalue(expr)
        else:
            return False, None

    # Expression value getters for particular AST node types.
    def _getvalue_expr_Constant(self, expr: ast.Constant) -> Any:
        """
        Value of a Constant expression.
        """
        return expr.value

    def _getvalue_expr_NameConstant(self, expr: ast.NameConstant) -> Any:
        """
        Value of a NameConstant expression.
        """
        return expr.value

    def _getvalue_expr_Num(self, expr: ast.Num) -> Any:
        """
        Value of a Num expression.
        """
        return expr.n

    def _getvalue_expr_Str(self, expr: ast.Str) -> Any:
        """
        Value of a Str expression.
        """
        return expr.s

    def _getvalue_expr_Bytes(self, expr: ast.Bytes) -> Any:
        """
        Value of a Bytes expression.
        """
        return expr.s

    def _getvalue_expr_Ellipsis(self, expr: ast.Ellipsis) -> Any:
        """
        Value of an Ellipsis expression.
        """
        return Ellipsis

    # Statement analyzers for particular AST node types.

    def _analyse_stmt_AnnAssign(
        self, statement: ast.AnnAssign, *, next: CFNode
    ) -> CFNode:
        """
        Analyse an annotated assignment statement.
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_Assert(self, statement: ast.Assert, *, next: CFNode) -> CFNode:
        """
        Analyse an assert statement.
        """
        test_is_constant, test_value = self._expression_as_constant(statement.test)

        branches: Dict[str, CFNode] = {}
        if test_is_constant:
            if test_value:
                branches.update(next=next)
            else:
                branches.update(error=self._raise)
        else:
            branches.update(next=next, error=self._raise)

        return self._ast_node(statement, **branches)

    def _analyse_stmt_Assign(self, statement: ast.Assign, *, next: CFNode) -> CFNode:
        """
        Analyse an assignment statement.
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_AsyncFor(
        self, statement: ast.AsyncFor, *, next: CFNode
    ) -> CFNode:
        """
        Analyse an async for statement.
        """
        return self._analyse_loop(statement, next=next)

    def _analyse_stmt_AsyncFunctionDef(
        self, statement: ast.AsyncFunctionDef, *, next: CFNode
    ) -> CFNode:
        """
        Analyse an async function (coroutine) definition.
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_AsyncWith(
        self, statement: ast.AsyncWith, *, next: CFNode
    ) -> CFNode:
        """
        Analyse an async with statement.
        """
        return self._analyse_with(statement, next=next)

    def _analyse_stmt_AugAssign(
        self, statement: ast.AugAssign, *, next: CFNode
    ) -> CFNode:
        """
        Analyse an augmented assignment statement.
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_Break(self, statement: ast.Break, *, next: CFNode) -> CFNode:
        """
        Analyse a break statement.
        """
        return self._ast_node(statement, next=self._context[_BREAK])

    def _analyse_stmt_ClassDef(
        self, statement: ast.ClassDef, *, next: CFNode
    ) -> CFNode:
        """
        Analyse a class definition.
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_Continue(
        self, statement: ast.Continue, *, next: CFNode
    ) -> CFNode:
        """
        Analyse a continue statement.
        """
        return self._ast_node(statement, next=self._context[_CONTINUE])

    def _analyse_stmt_Delete(self, statement: ast.Delete, *, next: CFNode) -> CFNode:
        """
        Analyse a del statement.
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_Expr(self, statement: ast.Expr, *, next: CFNode) -> CFNode:
        """
        Analyse an expression (used as a statement).
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_For(self, statement: ast.For, *, next: CFNode) -> CFNode:
        """
        Analyse a for statement.
        """
        return self._analyse_loop(statement, next=next)

    def _analyse_stmt_FunctionDef(
        self, statement: ast.FunctionDef, *, next: CFNode
    ) -> CFNode:
        """
        Analyse a function definition.
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_Global(self, statement: ast.Global, *, next: CFNode) -> CFNode:
        """
        Analyse a global statement
        """
        return self._ast_node(statement, next=next)

    def _analyse_stmt_If(self, statement: ast.If, *, next: CFNode) -> CFNode:
        """
        Analyse an if statement.
        """
        # Analyse both branches unconditionally: even if they're not reachable,
        # they still need to exist in the graph produced.
        if_branch = self._analyse_statements(statement.body, next=next)
        else_branch = self._analyse_statements(statement.orelse, next=next)

        # Analyse the condition, if a constant.
        branches: Dict[str, CFNode] = {}
        test_is_constant, test_value = self._expression_as_constant(statement.test)
        if test_is_constant:
            if test_value:
                branches.update(enter=if_branch)
            else:
                branches.update(else_=else_branch)
        else:
            branches.update(enter=if_branch, else_=else_branch, error=self._raise)

        return self._ast_node(statement, **branches)

    def _analyse_stmt_Import(self, statement: ast.Import, *, next: CFNode) -> CFNode:
        """
        Analyse an import statement.
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_ImportFrom(
        self, statement: ast.ImportFrom, *, next: CFNode
    ) -> CFNode:
        """
        Analyse a from ... import statement.
        """
        return self._ast_node(statement, next=next, error=self._raise)

    def _analyse_stmt_Nonlocal(
        self, statement: ast.Nonlocal, *, next: CFNode
    ) -> CFNode:
        """
        Analyse a nonlocal statement.
        """
        return self._ast_node(statement, next=next)

    def _analyse_stmt_Pass(self, statement: ast.Pass, *, next: CFNode) -> CFNode:
        """
        Analyse a pass statement.
        """
        return self._ast_node(statement, next=next)

    def _analyse_stmt_Raise(self, statement: ast.Raise, *, next: CFNode) -> CFNode:
        """
        Analyse a raise statement.
        """
        return self._ast_node(statement, error=self._raise)

    def _analyse_stmt_Return(self, statement: ast.Return, *, next: CFNode) -> CFNode:
        """
        Analyse a return statement.
        """
        if statement.value is None:
            nodes = dict(next=self._context[_LEAVE])
        else:
            nodes = dict(next=self._context[_RETURN])
            value_is_constant, _ = self._expression_as_constant(statement.value)
            if not value_is_constant:
                nodes.update(error=self._raise)
        return self._ast_node(statement, **nodes)

    def _analyse_stmt_Try(self, statement: ast.Try, *, next: CFNode) -> CFNode:
        """
        Analyse a try statement.
        """
        # To analyse a try-except-else-finally block, it's easier to think of
        # it as two separate pieces: it's equivalent to the try-except-else
        # piece, nested *inside* a try-finally. Analysis of the try-except-else
        # piece is fairly straightforward, and is handled by the
        # _analyse_try_except method.

        # The finally block can be entered by up to six different means,
        # depending on the context: from a return with-or-without value (if
        # inside a function); from a raise; from a break or continue (if within
        # a loop); or from simply completing the try-except-else part in the
        # usual way.
        #
        # For each of these different entrances to the finally block, the
        # target on *leaving* the finally block may be different. So we will
        # construct different paths in the control flow graph for each of the
        # possible leave targets for the finally block. (This is similar to the
        # way that Python >= 3.9 bytecode handles the finally.) We always
        # analyse the 'direct' path in which no raise, return, continue or
        # break occurs, even if that path is unreachable. For the other five
        # paths, we analyse them only if reachable. To do this, we adopt the
        # following approach:
        #
        # 1. Create dummy target nodes for the try-except-else, and analyse
        #    the try-except-else with those dummy nodes.
        # 2. For only the dummy nodes that are actually reached, construct
        #    the corresponding finally paths.
        # 3. Link up the graphs.

        finally_node = self._analyse_statements(statement.finalbody, next=next)

        dummy_nodes = {}
        target_nodes = {}
        for label, node in self._context.items():
            if node != next and node not in dummy_nodes:
                dummy_nodes[node] = self._dummy_node()
            target_nodes[label] = finally_node if node == next else dummy_nodes[node]

        with self._updated_context(**target_nodes):
            entry_node = self._analyse_try_except(statement, next=finally_node)

        # Remove dummy nodes that aren't reached; replace those that are with a
        # path through the finally branch.
        for end_node, dummy_node in dummy_nodes.items():
            if end_node == next or self._has_parents(dummy_node):
                self._graph.collapse_node(
                    dummy_node,
                    self._analyse_statements(statement.finalbody, next=end_node),
                )
            else:
                self._graph.remove_node(dummy_node)

        return entry_node

    def _analyse_stmt_While(self, statement: ast.While, *, next: CFNode) -> CFNode:
        """
        Analyse a while statement.
        """
        # Analyse the else branch.
        else_node = self._analyse_statements(statement.orelse, next=next)

        # Analyse the body.
        dummy_node = self._dummy_node()
        with self._updated_context(break_=next, continue_=dummy_node):
            body_node = self._analyse_statements(statement.body, next=dummy_node)

        # Analyse the condition, if a constant.
        branches: Dict[str, CFNode] = {}
        test_is_constant, test_value = self._expression_as_constant(statement.test)
        if test_is_constant:
            if test_value:
                branches.update(enter=body_node)
            else:
                branches.update(else_=else_node)
        else:
            branches.update(enter=body_node, else_=else_node, error=self._raise)

        loop_node = self._ast_node(statement, **branches)
        self._graph.collapse_node(dummy_node, loop_node)
        return loop_node

    def _analyse_stmt_With(self, statement: ast.With, *, next: CFNode) -> CFNode:
        """
        Analyse a with statement.
        """
        return self._analyse_with(statement, next=next)

    def analyse_class(self, ast_node: ast.ClassDef) -> CFAnalysis:
        """
        Construct a control flow graph for an ast.Class node.
        """
        leave_node = self._annotated_node("<leave>")
        raise_node = self._annotated_node("<raise>")

        with self._updated_context(raise_=raise_node):
            entry_node = self._analyse_statements(ast_node.body, next=leave_node)

        self._annotated_node("<start>", enter=entry_node)

        key_nodes = dict(entry_node=entry_node)
        if self._has_parents(leave_node):
            key_nodes.update(leave_node=leave_node)
        else:
            self._graph.remove_node(leave_node)
        if self._has_parents(raise_node):
            key_nodes.update(raise_node=raise_node)
        else:
            self._graph.remove_node(raise_node)

        return CFAnalysis(graph=self._graph, **key_nodes)

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
        leave_node = self._annotated_node("<leave>")
        raise_node = self._annotated_node("<raise>")
        return_node = self._annotated_node("<return>")

        with self._updated_context(
            leave=leave_node,
            raise_=raise_node,
            return_=return_node,
        ):
            entry_node = self._analyse_statements(ast_node.body, next=leave_node)

        # Make sure there's at least one reference to the first node.
        self._annotated_node("<start>", enter=entry_node)

        key_nodes = dict(entry_node=entry_node)
        if self._has_parents(leave_node):
            key_nodes.update(leave_node=leave_node)
        else:
            self._graph.remove_node(leave_node)
        if self._has_parents(raise_node):
            key_nodes.update(raise_node=raise_node)
        else:
            self._graph.remove_node(raise_node)
        if self._has_parents(return_node):
            key_nodes.update(return_node=return_node)
        else:
            self._graph.remove_node(return_node)

        return CFAnalysis(graph=self._graph, **key_nodes)

    def analyse_module(self, ast_node: ast.Module) -> CFAnalysis:
        """
        Construct a control flow graph for an ast.Module node.
        """
        leave_node = self._annotated_node("<leave>")
        raise_node = self._annotated_node("<raise>")

        with self._updated_context(raise_=raise_node):
            entry_node = self._analyse_statements(ast_node.body, next=leave_node)

        self._annotated_node("<start>", enter=entry_node)

        key_nodes = dict(entry_node=entry_node)
        if self._has_parents(leave_node):
            key_nodes.update(leave_node=leave_node)
        else:
            self._graph.remove_node(leave_node)
        if self._has_parents(raise_node):
            key_nodes.update(raise_node=raise_node)
        else:
            self._graph.remove_node(raise_node)

        return CFAnalysis(graph=self._graph, **key_nodes)
