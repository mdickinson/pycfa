"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""
import ast

# Constants used as edge and context labels.
RAISE = "raise"
RETURN = "return"
RETURN_VALUE = "return_value"
LEAVE = "leave"
NEXT = "next"
IF = "if_branch"
ELSE = "else_branch"
CONTINUE = "continue"
BREAK = "break"


# TODO: add links from CFNodes to the corresponding AST nodes.
# TODO: try/finally
# TODO: try/except/else


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


def analyse_statements(stmts, context):
    """
    Analyse control flow for a sequence of statements.

    Parameters
    ----------
    stmts : list of ast.stmt
        Statements to be analysed.
    context : mapping from str to CFNode
        Context in which these statements are being analysed.
    """
    # It's convenient to iterate over statements in reverse, creating
    # a linked list from the last element in the list backwards.

    head = context[LEAVE]
    for stmt in reversed(stmts):
        if isinstance(stmt, ast.Pass):
            stmt_node = CFNode({NEXT: head})
        elif isinstance(stmt, (ast.Expr, ast.Assign)):
            stmt_node = CFNode({RAISE: context[RAISE], NEXT: head})
        elif isinstance(stmt, ast.Return):
            if stmt.value is None:
                stmt_node = CFNode({RETURN: context[RETURN]})
            else:
                stmt_node = CFNode(
                    {
                        RAISE: context[RAISE],
                        RETURN_VALUE: context[RETURN_VALUE],
                    }
                )
        elif isinstance(stmt, ast.If):
            if_context = context.copy()
            if_context[LEAVE] = head

            stmt_node = CFNode(
                {
                    IF: analyse_statements(stmt.body, if_context),
                    ELSE: analyse_statements(stmt.orelse, if_context),
                    RAISE: context[RAISE],
                }
            )
        elif isinstance(stmt, (ast.For, ast.While)):
            else_context = context.copy()
            else_context[LEAVE] = head
            else_node = analyse_statements(stmt.orelse, else_context)

            loop_node = CFNode({RAISE: context[RAISE], ELSE: else_node})

            body_context = context.copy()
            body_context[LEAVE] = loop_node
            body_context[CONTINUE] = loop_node
            body_context[BREAK] = head
            body_node = analyse_statements(stmt.body, body_context)

            loop_node.add_edge(NEXT, body_node)
            stmt_node = loop_node
        elif isinstance(stmt, ast.Raise):
            stmt_node = CFNode({RAISE: context[RAISE]})
        elif isinstance(stmt, ast.Continue):
            stmt_node = CFNode({CONTINUE: context[CONTINUE]})
        elif isinstance(stmt, ast.Break):
            stmt_node = CFNode({BREAK: context[BREAK]})
        else:
            raise NotImplementedError("Unhandled stmt type", type(stmt))

        head = stmt_node

    return head


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
        LEAVE: CFNode(),  # node for leaving by falling off the end
    }
    enter = analyse_statements(ast_node.body, context)
    return context, enter
