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


class CFNode:
    """
    A node on the control flow graph.
    """

    def __init__(self, edges={}):
        # Outward edges for possible control flow transfer.
        self._out = {}
        for name, target in edges.items():
            self.add_edge(name, target)

    def add_edge(self, name, target):
        # For now, be careful about overwriting.
        if name in self._out:
            raise ValueError("An edge with that name already exists")
        self._out[name] = target

    @property
    def edge_names(self):
        # Names of outward edges, as a set.
        return set(self._out.keys())

    def target(self, edge_name):
        return self._out[edge_name]


def analyse_simple(statement: ast.stmt, context: dict) -> CFNode:
    """
    Analyse a statement not involving control flow.
    """
    return CFNode({RAISE: context[RAISE], NEXT: context[NEXT]})


def analyse_global_or_nonlocal(statement: ast.Global, context: dict) -> CFNode:
    """
    Analyse a global or nonlocal statement.
    """
    return context[NEXT]


def analyse_pass(statement: ast.Pass, context: dict) -> CFNode:
    """
    Analyse a pass statement.
    """
    return CFNode({NEXT: context[NEXT]})


def analyse_return(statement: ast.Return, context: dict) -> CFNode:
    """
    Analyse a return statement.
    """
    if statement.value is None:
        return CFNode({RETURN: context[RETURN]})
    else:
        return CFNode(
            {RAISE: context[RAISE], RETURN_VALUE: context[RETURN_VALUE]}
        )


def analyse_if(statement: ast.If, context: dict) -> CFNode:
    """
    Analyse an if statement.
    """
    return CFNode(
        {
            IF: analyse_statements(statement.body, context),
            ELSE: analyse_statements(statement.orelse, context),
            RAISE: context[RAISE],
        }
    )


def analyse_for_or_while(statement: ast.stmt, context: dict) -> CFNode:
    """
    Analyse a for or while statement.
    """
    loop_node = CFNode(
        {
            RAISE: context[RAISE],
            ELSE: analyse_statements(statement.orelse, context),
            # The target for the ENTER edge is created below.
        }
    )

    body_context = context.copy()
    body_context[BREAK] = context[NEXT]
    body_context[CONTINUE] = loop_node
    body_context[NEXT] = loop_node
    body_node = analyse_statements(statement.body, body_context)

    loop_node.add_edge(ENTER, body_node)
    return loop_node


def analyse_raise(statement: ast.Raise, context: dict) -> CFNode:
    """
    Analyse a raise statement.
    """
    return CFNode({RAISE: context[RAISE]})


def analyse_break(statement: ast.Break, context: dict) -> CFNode:
    """
    Analyse a break statement.
    """
    return CFNode({BREAK: context[BREAK]})


def analyse_continue(statement: ast.Continue, context: dict) -> CFNode:
    """
    Analyse a continue statement.
    """
    return CFNode({CONTINUE: context[CONTINUE]})


def _analyse_try_except_else(statement: ast.Try, context: dict) -> CFNode:
    """
    Analyse the try-except-else part of a try statement. The finally
    part is ignored, as though it weren't present.
    """
    # Process handlers, backwards: if the last handler doesn't match, raise.
    raise_node = context[RAISE]
    for handler in reversed(statement.handlers):
        match_node = analyse_statements(handler.body, context)
        if handler.type is None:
            # Bare except always matches, never raises.
            raise_node = match_node
        else:
            raise_node = CFNode(
                {
                    RAISE: context[RAISE],
                    MATCH: match_node,
                    NO_MATCH: raise_node,
                }
            )

    body_context = context.copy()
    body_context[RAISE] = raise_node
    body_context[NEXT] = analyse_statements(statement.orelse, context)
    return analyse_statements(statement.body, body_context)


def analyse_try(statement: ast.Try, context: dict) -> CFNode:
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
                statement.finalbody, finally_context
            )

    return _analyse_try_except_else(statement, try_except_else_context)


def analyse_with(statement: ast.With, context: dict) -> CFNode:
    """
    Analyse a with statement.
    """
    return CFNode(
        {
            ENTER: analyse_statements(statement.body, context),
            RAISE: context[RAISE],
        }
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


def analyse_statement(stmt: ast.stmt, context: dict) -> CFNode:
    """
    Analyse control flow for a single statement.
    """
    return analysers[type(stmt)](stmt, context)


def analyse_statements(stmts: list, context: dict) -> CFNode:
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
        next = analysers[type(stmt)](stmt, stmt_context)

    return next


def analyse_function(ast_node):
    """
    Parameters
    ----------
    ast_node : ast.FunctionDef

    Returns
    -------
    context : mapping from str to CFNode
        Context for the function, giving nodes for the entry point
        and the various exit points.
    """
    context = {
        RAISE: CFNode(),
        RETURN_VALUE: CFNode(),  # node for 'return <expr>'
        RETURN: CFNode(),  # node for plain valueless return
        NEXT: CFNode(),  # node for leaving by falling off the end
    }
    enter = analyse_statements(ast_node.body, context)
    return context, enter
