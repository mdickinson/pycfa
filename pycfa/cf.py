"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Union

from pycfa.cfgraph import CFGraph, CFNode

# Context labels
BREAK = "break"
CONTINUE = "continue"
RETURN = "return"
RETURN_VALUE = "return_value"
NEXTC = "nextc"
RAISEC = "raisec"
ENTERC = "enterc"

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


class CFAnalysis:
    """
    The control-flow analysis.

    This is a directed graph (not necessarily acyclic) with labelled edges.
    Most nodes will correspond directly to an AST statement.
    """

    _graph: CFGraph

    context: Context

    def __init__(self) -> None:
        self._graph = CFGraph()

        # We'll usually want some named nodes. (For example, for a graph
        # representing the control flow in a function, we'll want to mark the
        # entry and exit nodes.) Those go in the dictionary below.
        self.context = {}

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

    def new_node(
        self,
        edges: Dict[str, CFNode],
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
        node = CFNode(ast_node=ast_node, annotation=annotation)
        self._graph.add_node(node, edges)
        return node

    def _analyse_declaration(
        self, statement: Union[ast.Global, ast.Nonlocal], context: Context
    ) -> CFNode:
        """
        Analyse a declaration (a global or nonlocal statement).

        These statements are non-executable, so we don't create a node
        for them. Instead, we simply return the NEXT node from the context.
        """
        return context[NEXTC]

    def _analyse_generic(self, statement: ast.stmt, context: Context) -> CFNode:
        """
        Analyse a generic statement that doesn't affect control flow.
        """
        return self.new_node(
            {RAISE: context[RAISEC], NEXT: context[NEXTC]}, ast_node=statement
        )

    def _analyse_loop(
        self,
        statement: Union[ast.AsyncFor, ast.For, ast.While],
        context: Context,
    ) -> CFNode:
        """
        Analyse a loop statement (for or while).
        """
        dummy_node = self.new_node({}, annotation="<dummy>")

        body_context = context.copy()
        body_context[BREAK] = context[NEXTC]
        body_context[CONTINUE] = dummy_node
        body_context[NEXTC] = dummy_node
        body_node = self.analyse_statements(statement.body, body_context)

        loop_node = self.new_node(
            {
                RAISE: context[RAISEC],
                ELSE: self.analyse_statements(statement.orelse, context),
                ENTER: body_node,
            },
            ast_node=statement,
        )

        self._graph.collapse_node(dummy_node, loop_node)

        return loop_node

    def _analyse_try_except_else(self, statement: ast.Try, context: Context) -> CFNode:
        """
        Analyse the try-except-else part of a try statement. The finally
        part is ignored, as though it weren't present.
        """
        # Process handlers backwards; raise if the last handler doesn't match.
        raise_node = context[RAISEC]
        for handler in reversed(statement.handlers):
            match_node = self.analyse_statements(handler.body, context)
            if handler.type is None:
                # Bare except always matches, never raises.
                raise_node = match_node
            else:
                raise_node = self.new_node(
                    {
                        RAISE: context[RAISEC],
                        ENTER: match_node,
                        ELSE: raise_node,
                    },
                    ast_node=handler.type,
                )

        body_context = context.copy()
        body_context[RAISEC] = raise_node
        body_context[NEXTC] = self.analyse_statements(statement.orelse, context)
        body_node = self.analyse_statements(statement.body, body_context)

        return self.new_node({NEXT: body_node}, ast_node=statement)

    def _analyse_with(
        self, statement: Union[ast.AsyncWith, ast.With], context: Context
    ) -> CFNode:
        """
        Analyse a with or async with statement.
        """
        return self.new_node(
            {
                ENTER: self.analyse_statements(statement.body, context),
                RAISE: context[RAISEC],
            },
            ast_node=statement,
        )

    def analyse_AnnAssign(self, statement: ast.AnnAssign, context: Context) -> CFNode:
        """
        Analyse an annotated assignment statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_Assert(self, statement: ast.Assert, context: Context) -> CFNode:
        """
        Analyse an assert statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_Assign(self, statement: ast.Assign, context: Context) -> CFNode:
        """
        Analyse an assignment statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_AsyncFor(self, statement: ast.AsyncFor, context: Context) -> CFNode:
        """
        Analyse an async for statement.
        """
        return self._analyse_loop(statement, context)

    def analyse_AsyncFunctionDef(
        self, statement: ast.AsyncFunctionDef, context: Context
    ) -> CFNode:
        """
        Analyse an async function (coroutine) definition.
        """
        return self._analyse_generic(statement, context)

    def analyse_AsyncWith(self, statement: ast.AsyncWith, context: Context) -> CFNode:
        """
        Analyse an async with statement.
        """
        return self._analyse_with(statement, context)

    def analyse_AugAssign(self, statement: ast.AugAssign, context: Context) -> CFNode:
        """
        Analyse an augmented assignment statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_Break(self, statement: ast.Break, context: Context) -> CFNode:
        """
        Analyse a break statement.
        """
        return self.new_node({NEXT: context[BREAK]}, ast_node=statement)

    def analyse_ClassDef(self, statement: ast.ClassDef, context: Context) -> CFNode:
        """
        Analyse a class definition.
        """
        return self._analyse_generic(statement, context)

    def analyse_Continue(self, statement: ast.Continue, context: Context) -> CFNode:
        """
        Analyse a continue statement.
        """
        return self.new_node({NEXT: context[CONTINUE]}, ast_node=statement)

    def analyse_Delete(self, statement: ast.Delete, context: Context) -> CFNode:
        """
        Analyse a del statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_Expr(self, statement: ast.Expr, context: Context) -> CFNode:
        """
        Analyse an expression (used as a statement).
        """
        return self._analyse_generic(statement, context)

    def analyse_For(self, statement: ast.For, context: Context) -> CFNode:
        """
        Analyse a for statement.
        """
        return self._analyse_loop(statement, context)

    def analyse_FunctionDef(
        self, statement: ast.FunctionDef, context: Context
    ) -> CFNode:
        """
        Analyse a function definition.
        """
        return self._analyse_generic(statement, context)

    def analyse_Global(self, statement: ast.Global, context: Context) -> CFNode:
        """
        Analyse a global statement
        """
        return self._analyse_declaration(statement, context)

    def analyse_If(self, statement: ast.If, context: Context) -> CFNode:
        """
        Analyse an if statement.
        """
        return self.new_node(
            {
                ENTER: self.analyse_statements(statement.body, context),
                ELSE: self.analyse_statements(statement.orelse, context),
                RAISE: context[RAISEC],
            },
            ast_node=statement,
        )

    def analyse_Import(self, statement: ast.Import, context: Context) -> CFNode:
        """
        Analyse an import statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_ImportFrom(self, statement: ast.ImportFrom, context: Context) -> CFNode:
        """
        Analyse a from ... import statement.
        """
        return self._analyse_generic(statement, context)

    def analyse_Nonlocal(self, statement: ast.Nonlocal, context: Context) -> CFNode:
        """
        Analyse a nonlocal statement
        """
        return self._analyse_declaration(statement, context)

    def analyse_Pass(self, statement: ast.Pass, context: Context) -> CFNode:
        """
        Analyse a pass statement.
        """
        return self.new_node({NEXT: context[NEXTC]}, ast_node=statement)

    def analyse_Raise(self, statement: ast.Raise, context: Context) -> CFNode:
        """
        Analyse a raise statement.
        """
        return self.new_node({RAISE: context[RAISEC]}, ast_node=statement)

    def analyse_Return(self, statement: ast.Return, context: Context) -> CFNode:
        """
        Analyse a return statement.
        """
        if statement.value is None:
            return self.new_node({NEXT: context[RETURN]}, ast_node=statement)
        else:
            return self.new_node(
                {NEXT: context[RETURN_VALUE], RAISE: context[RAISEC]},
                ast_node=statement,
            )

    def analyse_Try(self, statement: ast.Try, context: Context) -> CFNode:
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
        dummy_nodes = {
            node: self.new_node({}, annotation="<dummy>")
            for node in set(context.values())
        }

        # Analyse the try-except-else part of the statement using those dummy
        # nodes.
        try_except_else_context = {
            key: dummy_nodes[node] for key, node in context.items()
        }
        entry_node = self._analyse_try_except_else(statement, try_except_else_context)

        # Now iterate through the dummy nodes. For those that aren't reached,
        # remove them. For those that are, replace with the corresponding
        # finally code. Note that there will always be at least one of the
        # dummy nodes reachable from the try-except-else, so we'll always
        # analyse the finally code at least once.

        for end_node, dummy_node in dummy_nodes.items():
            if self._graph.edges_to(dummy_node):
                # Dummy node is reachable from the try-except-else.
                finally_context = context.copy()
                finally_context[NEXTC] = end_node
                finally_node = self.analyse_statements(
                    statement.finalbody, finally_context
                )

                self._graph.collapse_node(dummy_node, finally_node)
            else:
                self._graph.remove_node(dummy_node)

        return entry_node

    def analyse_While(self, statement: ast.While, context: Context) -> CFNode:
        """
        Analyse a while statement
        """
        return self._analyse_loop(statement, context)

    def analyse_With(self, statement: ast.With, context: Context) -> CFNode:
        """
        Analyse a with statement.
        """
        return self._analyse_with(statement, context)

    def analyse_statements(
        self, statements: List[ast.stmt], context: Context
    ) -> CFNode:
        """
        Analyse a sequence of statements.
        """
        next = context[NEXTC]
        for statement in reversed(statements):
            statement_context = context.copy()
            statement_context[NEXTC] = next

            method_name = "analyse_" + type(statement).__name__
            analyser = getattr(self, method_name)
            next = analyser(statement, statement_context)

        return next

    @classmethod
    def from_class(cls, ast_node: ast.ClassDef) -> CFAnalysis:
        """
        Construct a control flow graph for an ast.Module node.
        """
        self = cls()

        leave_node = self.new_node({}, annotation="<leave>")
        raise_node = self.new_node({}, annotation="<raise>")

        body_context = {
            NEXTC: leave_node,
            RAISEC: raise_node,
        }
        enter_node = self.analyse_statements(ast_node.body, body_context)

        self.context = {
            ENTERC: enter_node,
            NEXTC: leave_node,
            RAISEC: raise_node,
        }
        return self

    @classmethod
    def from_function(
        cls, ast_node: Union[ast.AsyncFunctionDef, ast.FunctionDef]
    ) -> CFAnalysis:
        """
        Construct a control flow graph for a function or coroutine AST node.

        Parameters
        ----------
        ast_node : ast.FunctionDef or ast.AsyncFunctionDef
            Function node in the ast tree of the code being analysed.
        """
        self = cls()

        # Node for returns without an explicit return value.
        return_node = self.new_node({}, annotation="<return-without-value>")
        # Node for returns *with* an explicit return value (which could
        # be None).
        return_value_node = self.new_node({}, annotation="<return-with-value>")
        # Node for exit via raise.
        raise_node = self.new_node({}, annotation="<raise>")

        body_context = {
            NEXTC: return_node,
            RAISEC: raise_node,
            RETURN: return_node,
            RETURN_VALUE: return_value_node,
        }
        enter_node = self.analyse_statements(ast_node.body, body_context)

        self.context = {
            ENTERC: enter_node,
            RAISEC: raise_node,
            RETURN: return_node,
            RETURN_VALUE: return_value_node,
        }
        return self

    @classmethod
    def from_module(cls, ast_node: ast.Module) -> CFAnalysis:
        """
        Construct a control flow graph for an ast.Module node.
        """
        self = cls()

        leave_node = self.new_node({}, annotation="<leave>")
        raise_node = self.new_node({}, annotation="<raise>")

        body_context = {
            NEXTC: leave_node,
            RAISEC: raise_node,
        }
        enter_node = self.analyse_statements(ast_node.body, body_context)

        self.context = {
            ENTERC: enter_node,
            NEXTC: leave_node,
            RAISEC: raise_node,
        }
        return self
