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


class CFNode:
    """
    A node on the control flow graph.
    """

    def __init__(self, edges, graph):
        self.graph = graph

        self.graph.add_node(self)
        for name, target in edges.items():
            self.graph.add_edge(self, name, target)

    def target(self, edge_name):
        return self.graph.edges[self][edge_name]


def analyse_simple(
    statement: ast.stmt, context: dict, graph: CFGraph
) -> CFNode:
    """
    Analyse a statement not involving control flow.
    """
    return CFNode({RAISE: context[RAISE], NEXT: context[NEXT]}, graph)


def analyse_global_or_nonlocal(
    statement: ast.Global, context: dict, graph: CFGraph
) -> CFNode:
    """
    Analyse a global or nonlocal statement.
    """
    return context[NEXT]


def analyse_pass(statement: ast.Pass, context: dict, graph: CFGraph) -> CFNode:
    """
    Analyse a pass statement.
    """
    return CFNode({NEXT: context[NEXT]}, graph)


def analyse_return(
    statement: ast.Return, context: dict, graph: CFGraph
) -> CFNode:
    """
    Analyse a return statement.
    """
    if statement.value is None:
        return CFNode({RETURN: context[RETURN]}, graph)
    else:
        return CFNode(
            {RAISE: context[RAISE], RETURN_VALUE: context[RETURN_VALUE]}, graph
        )


def analyse_if(statement: ast.If, context: dict, graph: CFGraph) -> CFNode:
    """
    Analyse an if statement.
    """
    return CFNode(
        {
            IF: analyse_statements(statement.body, context, graph),
            ELSE: analyse_statements(statement.orelse, context, graph),
            RAISE: context[RAISE],
        },
        graph,
    )


def analyse_for_or_while(
    statement: ast.stmt, context: dict, graph: CFGraph
) -> CFNode:
    """
    Analyse a for or while statement.
    """
    loop_node = CFNode(
        {
            RAISE: context[RAISE],
            ELSE: analyse_statements(statement.orelse, context, graph),
            # The target for the ENTER edge is created below.
        },
        graph,
    )

    body_context = context.copy()
    body_context[BREAK] = context[NEXT]
    body_context[CONTINUE] = loop_node
    body_context[NEXT] = loop_node
    body_node = analyse_statements(statement.body, body_context, graph)

    graph.add_edge(loop_node, ENTER, body_node)
    return loop_node


def analyse_raise(
    statement: ast.Raise, context: dict, graph: CFGraph
) -> CFNode:
    """
    Analyse a raise statement.
    """
    return CFNode({RAISE: context[RAISE]}, graph)


def analyse_break(
    statement: ast.Break, context: dict, graph: CFGraph
) -> CFNode:
    """
    Analyse a break statement.
    """
    return CFNode({BREAK: context[BREAK]}, graph)


def analyse_continue(
    statement: ast.Continue, context: dict, graph: CFGraph
) -> CFNode:
    """
    Analyse a continue statement.
    """
    return CFNode({CONTINUE: context[CONTINUE]}, graph)


def _analyse_try_except_else(
    statement: ast.Try, context: dict, graph: CFGraph
) -> CFNode:
    """
    Analyse the try-except-else part of a try statement. The finally
    part is ignored, as though it weren't present.
    """
    # Process handlers, backwards: if the last handler doesn't match, raise.
    raise_node = context[RAISE]
    for handler in reversed(statement.handlers):
        match_node = analyse_statements(handler.body, context, graph)
        if handler.type is None:
            # Bare except always matches, never raises.
            raise_node = match_node
        else:
            raise_node = CFNode(
                {
                    RAISE: context[RAISE],
                    MATCH: match_node,
                    NO_MATCH: raise_node,
                },
                graph,
            )

    body_context = context.copy()
    body_context[RAISE] = raise_node
    body_context[NEXT] = analyse_statements(statement.orelse, context, graph)
    body_node = analyse_statements(statement.body, body_context, graph)

    return CFNode({ENTER: body_node}, graph)


def analyse_try(statement: ast.Try, context: dict, graph: CFGraph) -> CFNode:
    """
    Analyse a try statement.
    """
    try_except_else_context = context.copy()

    # We process the finally block several times, with a different NEXT target
    # each time. A break / continue / raise / return in any of the try, except
    # or else blocks will transfer control to the finally block, and then on
    # leaving the finally block, will transfer control back to whereever it
    # would have gone without the finally. Similarly, leaving the
    # try/except/else compound normally again transfers control to the finally
    # block, and then on leaving the finally, transfers control to wherever we
    # would have gone if the finally were not present.

    for node_type in [BREAK, CONTINUE, NEXT, RAISE, RETURN, RETURN_VALUE]:
        if node_type in context:
            finally_context = context.copy()
            finally_context[NEXT] = context[node_type]
            try_except_else_context[node_type] = analyse_statements(
                statement.finalbody, finally_context, graph
            )

    return _analyse_try_except_else(statement, try_except_else_context, graph)


def analyse_with(statement: ast.With, context: dict, graph: CFGraph) -> CFNode:
    """
    Analyse a with statement.
    """
    return CFNode(
        {
            ENTER: analyse_statements(statement.body, context, graph),
            RAISE: context[RAISE],
        },
        graph,
    )


# Mapping from statement types to the functions that can analyse them.

analysers = {
    ast.Assert: analyse_simple,
    ast.Assign: analyse_simple,
    ast.AugAssign: analyse_simple,
    ast.Break: analyse_break,
    ast.ClassDef: analyse_simple,
    ast.Continue: analyse_continue,
    ast.Delete: analyse_simple,
    ast.Expr: analyse_simple,
    ast.For: analyse_for_or_while,
    ast.Global: analyse_global_or_nonlocal,
    ast.FunctionDef: analyse_simple,
    ast.If: analyse_if,
    ast.Import: analyse_simple,
    ast.ImportFrom: analyse_simple,
    ast.Nonlocal: analyse_global_or_nonlocal,
    ast.Pass: analyse_pass,
    ast.Raise: analyse_raise,
    ast.Return: analyse_return,
    ast.Try: analyse_try,
    ast.While: analyse_for_or_while,
    ast.With: analyse_with,
}


def analyse_statements(stmts: list, context: dict, graph: CFGraph) -> CFNode:
    """
    Analyse control flow for a sequence of statements.

    Parameters
    ----------
    stmts : list of ast.stmt
        Statements to be analysed.
    context : mapping from str to CFNode
        Context in which these statements are being analysed.
        Should always provide at least NEXT and RAISE nodes. Other nodes
        may be provided, depending on context: RETURN and RETURN_VALUE
        if within a function context, and BREAK and CONTINUE if within
        a loop context.

    Returns
    -------
    CFNode
        Node corresponding to the first statement in the statement list.
        (If the statement list is empty, the NEXT node from the context
        will be returned.)
    """
    # It's convenient to iterate over statements in reverse, creating
    # a linked list from the last element in the list backwards.

    next = context[NEXT]
    for stmt in reversed(stmts):
        stmt_context = context.copy()
        stmt_context[NEXT] = next
        next = analysers[type(stmt)](stmt, stmt_context, graph)

    return next


def analyse_function(ast_node, graph):
    """
    Parameters
    ----------
    ast_node : ast.FunctionDef
    graph : CFGraph

    Returns
    -------
    context : mapping from str to CFNode
        Context for the function, giving nodes for the entry point
        and the various exit points.
    """
    context = {
        RAISE: CFNode({}, graph),
        RETURN_VALUE: CFNode({}, graph),  # node for 'return <expr>'
        RETURN: CFNode({}, graph),  # node for plain valueless return
        NEXT: CFNode({}, graph),  # node for leaving by falling off the end
    }
    enter = analyse_statements(ast_node.body, context, graph)
    return context, enter
