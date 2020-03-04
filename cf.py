"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""
import ast

# Constants used as both edge and context labels.
BREAK = "break"
CONTINUE = "continue"
NEXT = "next"
RAISE = "raise"
RETURN = "return"
RETURN_VALUE = "return_value"

# Constants used only as edge labels.
IF = "if"
ELSE = "else"
ENTER = "enter"
MATCH = "match"
NO_MATCH = "no_match"


# Mapping from statement types to the names of the methods that can analyse
# them.
ANALYSERS = {
    ast.Assert: "analyse_simple",
    ast.Assign: "analyse_simple",
    ast.AugAssign: "analyse_simple",
    ast.Break: "analyse_break",
    ast.ClassDef: "analyse_simple",
    ast.Continue: "analyse_continue",
    ast.Delete: "analyse_simple",
    ast.Expr: "analyse_simple",
    ast.For: "analyse_for_or_while",
    ast.Global: "analyse_global_or_nonlocal",
    ast.FunctionDef: "analyse_simple",
    ast.If: "analyse_if",
    ast.Import: "analyse_simple",
    ast.ImportFrom: "analyse_simple",
    ast.Nonlocal: "analyse_global_or_nonlocal",
    ast.Pass: "analyse_pass",
    ast.Raise: "analyse_raise",
    ast.Return: "analyse_return",
    ast.Try: "analyse_try",
    ast.While: "analyse_for_or_while",
    ast.With: "analyse_with",
}


class CFNode:
    """
    A node on the control flow graph.
    """


class CFGraph:
    """
    The control-flow graph.

    This is a directed graph (not necessarily acyclic) with labelled edges.
    Most nodes will correspond directly to an AST statement.
    """

    def __init__(self):
        # Representation: nodes form a set; for each node, edges[node]
        # is a mapping from labels to nodes.
        self.nodes = set()
        self.edges = {}

    def add_node(self, node):
        assert node not in self.nodes
        self.nodes.add(node)
        self.edges[node] = {}

    def add_edge(self, source, label, target):
        source_edges = self.edges[source]

        assert label not in source_edges
        source_edges[label] = target

    def cfnode(self, edges):
        """
        Create a new control-flow node and add it to the graph.

        Returns the newly-created node.
        """
        node = CFNode()
        self.add_node(node)
        for name, target in edges.items():
            self.add_edge(node, name, target)
        return node

    def analyse_break(self, statement: ast.Break, context: dict) -> CFNode:
        """
        Analyse a break statement.
        """
        return self.cfnode({BREAK: context[BREAK]})

    def analyse_continue(
        self, statement: ast.Continue, context: dict
    ) -> CFNode:
        """
        Analyse a continue statement.
        """
        return self.cfnode({CONTINUE: context[CONTINUE]})

    def analyse_for_or_while(
        self, statement: ast.stmt, context: dict
    ) -> CFNode:
        """
        Analyse a for or while statement.
        """
        loop_node = self.cfnode(
            {
                RAISE: context[RAISE],
                ELSE: self.analyse_statements(statement.orelse, context),
                # The target for the ENTER edge is created below.
            }
        )

        body_context = context.copy()
        body_context[BREAK] = context[NEXT]
        body_context[CONTINUE] = loop_node
        body_context[NEXT] = loop_node
        body_node = self.analyse_statements(statement.body, body_context)

        self.add_edge(loop_node, ENTER, body_node)
        return loop_node

    def analyse_global_or_nonlocal(
        self, statement: ast.Global, context: dict
    ) -> CFNode:
        """
        Analyse a global or nonlocal statement.

        These statements are non-executable, so we don't create a node
        for them. Instead, we simply return the NEXT node from the context.
        """
        return context[NEXT]

    def analyse_if(self, statement: ast.If, context: dict) -> CFNode:
        """
        Analyse an if statement.
        """
        return self.cfnode(
            {
                IF: self.analyse_statements(statement.body, context),
                ELSE: self.analyse_statements(statement.orelse, context),
                RAISE: context[RAISE],
            }
        )

    def analyse_pass(self, statement: ast.Pass, context: dict) -> CFNode:
        """
        Analyse a pass statement.
        """
        return self.cfnode({NEXT: context[NEXT]})

    def analyse_raise(self, statement: ast.Raise, context: dict) -> CFNode:
        """
        Analyse a raise statement.
        """
        return self.cfnode({RAISE: context[RAISE]})

    def analyse_return(self, statement: ast.Return, context: dict) -> CFNode:
        """
        Analyse a return statement.
        """
        if statement.value is None:
            return self.cfnode({RETURN: context[RETURN]})
        else:
            return self.cfnode(
                {RAISE: context[RAISE], RETURN_VALUE: context[RETURN_VALUE]},
            )

    def analyse_simple(self, statement: ast.stmt, context: dict) -> CFNode:
        """
        Analyse a statement not involving control flow.
        """
        return self.cfnode({RAISE: context[RAISE], NEXT: context[NEXT]})

    def _analyse_try_except_else(
        self, statement: ast.Try, context: dict
    ) -> CFNode:
        """
        Analyse the try-except-else part of a try statement. The finally
        part is ignored, as though it weren't present.
        """
        # Process handlers backwards; raise if the last handler doesn't match.
        raise_node = context[RAISE]
        for handler in reversed(statement.handlers):
            match_node = self.analyse_statements(handler.body, context)
            if handler.type is None:
                # Bare except always matches, never raises.
                raise_node = match_node
            else:
                raise_node = self.cfnode(
                    {
                        RAISE: context[RAISE],
                        MATCH: match_node,
                        NO_MATCH: raise_node,
                    }
                )

        body_context = context.copy()
        body_context[RAISE] = raise_node
        body_context[NEXT] = self.analyse_statements(statement.orelse, context)
        body_node = self.analyse_statements(statement.body, body_context)

        return self.cfnode({ENTER: body_node})

    def analyse_try(self, statement: ast.Try, context: dict) -> CFNode:
        """
        Analyse a try statement.
        """
        try_except_else_context = context.copy()

        # We process the finally block several times, with a different NEXT
        # target each time. A break / continue / raise / return in any of the
        # try, except or else blocks will transfer control to the finally
        # block, and then on leaving the finally block, will transfer control
        # back to whereever it would have gone without the finally. Similarly,
        # leaving the try/except/else compound normally again transfers control
        # to the finally block, and then on leaving the finally, transfers
        # control to wherever we would have gone if the finally were not
        # present.

        for node_type in [BREAK, CONTINUE, NEXT, RAISE, RETURN, RETURN_VALUE]:
            if node_type in context:
                finally_context = context.copy()
                finally_context[NEXT] = context[node_type]
                try_except_else_context[node_type] = self.analyse_statements(
                    statement.finalbody, finally_context
                )

        return self._analyse_try_except_else(
            statement, try_except_else_context
        )

    def analyse_with(self, statement: ast.With, context: dict) -> CFNode:
        """
        Analyse a with statement.
        """
        return self.cfnode(
            {
                ENTER: self.analyse_statements(statement.body, context),
                RAISE: context[RAISE],
            }
        )

    def analyse_statements(self, statements: list, context: dict) -> CFNode:
        """
        Analyse a sequence of statements.
        """
        next = context[NEXT]
        for statement in reversed(statements):
            statement_context = context.copy()
            statement_context[NEXT] = next
            analyser = getattr(self, ANALYSERS[type(statement)])
            next = analyser(statement, statement_context)

        return next


def analyse_function(ast_node):
    """
    Parameters
    ----------
    ast_node : ast.FunctionDef

    Returns
    -------
    graph : CFGraph
        Control flow graph for the function.
    context : mapping from str to CFNode
        Context for the function, giving nodes for the entry point
        and the various exit points.
    """
    graph = CFGraph()
    context = {
        RAISE: graph.cfnode({}),
        RETURN_VALUE: graph.cfnode({}),  # node for 'return <expr>'
        RETURN: graph.cfnode({}),  # node for plain valueless return
        NEXT: graph.cfnode({}),  # node for leaving by falling off the end
    }
    context[ENTER] = graph.analyse_statements(ast_node.body, context)
    return graph, context
