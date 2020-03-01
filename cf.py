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
MATCH = "match"  # match for exception clause
NO_MATCH = "no_match"  # failed match for exception clause


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
        elif isinstance(stmt, ast.Try):

            # Process the final clause 4 times, for the 4 different
            # situations in which it can be invoked.

            # 1st case: leave the finally block normally
            finally_leave_context = context.copy()
            finally_leave_context[LEAVE] = head
            finally_leave_node = analyse_statements(
                stmt.finalbody, finally_leave_context)

            # 2nd case: on leaving the finally block, re-raise
            # the exception that caused transfer to finally_raise_context
            finally_raise_context = context.copy()
            finally_raise_context[LEAVE] = context[RAISE]
            finally_raise_node = analyse_statements(
                stmt.finalbody, finally_raise_context)

            # 3rd case: on leaving the finally block, return a value
            # XXX We shouldn't create this if the context has no RETURN_VALUE.
            finally_return_value_context = context.copy()
            finally_return_value_context[LEAVE] = context[RETURN_VALUE]
            finally_return_value_node = analyse_statements(
                stmt.finalbody, finally_return_value_context)

            # 4th case: on leaving the finally block, return.
            # XXX We shouldn't create this if the context has no RETURN.
            finally_return_context = context.copy()
            finally_return_context[LEAVE] = context[RETURN]
            finally_return_node = analyse_statements(
                stmt.finalbody, finally_return_context)

            handler_context = context.copy()
            handler_context[LEAVE] = finally_leave_node

            # XXX test case of return in else or except clause
            handler_context[RETURN] = finally_return_node
            handler_context[RETURN_VALUE] = finally_return_value_node
            handler_context[RAISE] = finally_raise_node

            next_handler = finally_raise_node
            for handler in reversed(stmt.handlers):
                match_node = analyse_statements(handler.body, handler_context)
                if handler.type is None:
                    handler_node = CFNode({MATCH: match_node})
                else:
                    handler_node = CFNode(
                        {
                            RAISE: finally_raise_node,
                            MATCH: match_node,
                            NO_MATCH: next_handler,
                        }
                    )
                next_handler = handler_node

            else_handler = analyse_statements(stmt.orelse, handler_context)

            body_context = context.copy()
            body_context[RAISE] = next_handler
            body_context[LEAVE] = else_handler

            # XXX coverage for case where RETURN is not in context.
            body_context[RETURN] = finally_return_node
            # XXX coverage for case where RETURN_VALUE is not in context.
            body_context[RETURN_VALUE] = finally_return_value_node

            body_node = analyse_statements(stmt.body, body_context)
            stmt_node = body_node
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
