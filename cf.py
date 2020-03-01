"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""
import ast
import unittest

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
# TODO: Rename _add_edge back to add_edge
# TODO: test raise
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
            self._add_edge(name, target)

    def _add_edge(self, name, target):
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
    # context should _always_ have 'raise' and 'leave' entries
    # if there's a function context, may also have RETURN and
    # RETURN_VALUE entries. If there's a loop context, may
    # have additional entries for use with 'break' and 'continue'.

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
            # Inherit most of the context from the parent, but patch
            # the LEAVE entry.
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

            loop_node._add_edge(NEXT, body_node)
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


class TestCFAnalyser(unittest.TestCase):
    def test_analyse_noop_function(self):
        code = """\
def f():
    pass
"""
        function_context, pass_node = self._function_context(code)

        self.assertEqual(pass_node.edge_names, {NEXT})
        self.assertIs(pass_node.target(NEXT), function_context[LEAVE])

    def test_analyse_single_expr_statement(self):
        code = """\
def f():
    do_something()
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, {NEXT, RAISE})
        self.assertEqual(stmt_node.target(NEXT), function_context[LEAVE])
        self.assertEqual(stmt_node.target(RAISE), function_context[RAISE])

    def test_analyse_assign(self):
        code = """\
def f():
    a = 123
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, {NEXT, RAISE})
        self.assertEqual(stmt_node.target(NEXT), function_context[LEAVE])
        self.assertEqual(stmt_node.target(RAISE), function_context[RAISE])

    def test_analyse_multiple_statements(self):
        code = """\
def f():
    do_something()
    do_something_else()
"""
        function_context, stmt1_node = self._function_context(code)

        self.assertEqual(stmt1_node.edge_names, {NEXT, RAISE})
        self.assertEqual(stmt1_node.target(RAISE), function_context[RAISE])

        stmt2_node = stmt1_node.target(NEXT)
        self.assertEqual(stmt2_node.edge_names, {NEXT, RAISE})
        self.assertEqual(stmt2_node.target(NEXT), function_context[LEAVE])
        self.assertEqual(stmt2_node.target(RAISE), function_context[RAISE])

    def test_return_with_no_value(self):
        code = """\
def f():
    return
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, {RETURN})
        self.assertEqual(stmt_node.target(RETURN), function_context[RETURN])

    def test_return_with_value(self):
        code = """\
def f():
    return None
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, {RAISE, RETURN_VALUE})
        self.assertEqual(stmt_node.target(RAISE), function_context[RAISE])
        self.assertEqual(
            stmt_node.target(RETURN_VALUE), function_context[RETURN_VALUE]
        )

    def test_raise(self):
        code = """\
def f():
    raise TypeError("don't call me")
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, {RAISE})
        self.assertEqual(stmt_node.target(RAISE), function_context[RAISE])

    def test_if(self):
        code = """\
def f():
    if condition:
        a = 123
"""
        function_context, if_node = self._function_context(code)

        self.assertEqual(if_node.edge_names, {ELSE, IF, RAISE})
        self.assertEqual(if_node.target(RAISE), function_context[RAISE])

        if_branch = if_node.target(IF)
        self.assertEqual(if_branch.edge_names, {NEXT, RAISE})
        self.assertEqual(if_branch.target(RAISE), function_context[RAISE])
        self.assertEqual(if_branch.target(NEXT), function_context[LEAVE])
        self.assertEqual(if_node.target(ELSE), function_context[LEAVE])

    def test_if_else(self):
        code = """\
def f():
    if condition:
        a = 123
    else:
        b = 456
"""
        function_context, if_node = self._function_context(code)

        self.assertEqual(if_node.edge_names, {ELSE, IF, RAISE})
        self.assertEqual(if_node.target(RAISE), function_context[RAISE])

        if_branch = if_node.target(IF)
        self.assertEqual(if_branch.edge_names, {NEXT, RAISE})
        self.assertEqual(if_branch.target(RAISE), function_context[RAISE])
        self.assertEqual(if_branch.target(NEXT), function_context[LEAVE])

        else_branch = if_node.target(ELSE)
        self.assertEqual(else_branch.edge_names, {NEXT, RAISE})
        self.assertEqual(else_branch.target(RAISE), function_context[RAISE])
        self.assertEqual(else_branch.target(NEXT), function_context[LEAVE])

    def test_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return 123
    else:
        return 456
"""
        function_context, if_node = self._function_context(code)
        self.assertEqual(
            if_node.target(IF).edge_names, {RAISE, RETURN_VALUE},
        )
        self.assertEqual(
            if_node.target(IF).target(RETURN_VALUE),
            function_context[RETURN_VALUE],
        )
        self.assertEqual(
            if_node.target(IF).target(RAISE), function_context[RAISE],
        )

        self.assertEqual(
            if_node.target(ELSE).edge_names, {RAISE, RETURN_VALUE},
        )
        self.assertEqual(
            if_node.target(ELSE).target(RETURN_VALUE),
            function_context[RETURN_VALUE],
        )
        self.assertEqual(
            if_node.target(ELSE).target(RAISE), function_context[RAISE],
        )

    def test_plain_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return
    else:
        return
"""
        function_context, if_node = self._function_context(code)
        self.assertEqual(
            if_node.target(IF).edge_names, {RETURN},
        )
        self.assertEqual(
            if_node.target(IF).target(RETURN), function_context[RETURN],
        )

        self.assertEqual(
            if_node.target(ELSE).edge_names, {RETURN},
        )
        self.assertEqual(
            if_node.target(ELSE).target(RETURN), function_context[RETURN],
        )

    def test_unreachable_statements(self):
        code = """\
def f():
    do_something()
    return
    do_something_else()
"""
        function_context, stmt1_node = self._function_context(code)

        self.assertEqual(stmt1_node.edge_names, {NEXT, RAISE})
        self.assertEqual(stmt1_node.target(RAISE), function_context[RAISE])

        stmt2_node = stmt1_node.target(NEXT)
        self.assertEqual(stmt2_node.edge_names, {RETURN})
        self.assertEqual(stmt2_node.target(RETURN), function_context[RETURN])

    def test_while(self):
        code = """\
def f():
    while some_condition:
        do_something()
"""
        function_context, while_node = self._function_context(code)

        self.assertEqual(while_node.edge_names, {ELSE, NEXT, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])
        self.assertEqual(while_node.target(ELSE), function_context[LEAVE])

        body_node = while_node.target(NEXT)
        self.assertEqual(body_node.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

    def test_while_else(self):
        code = """\
def f():
    while some_condition:
        do_something()
    else:
        do_no_break_stuff()
"""
        function_context, while_node = self._function_context(code)

        self.assertEqual(while_node.edge_names, {ELSE, NEXT, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])

        body_node = while_node.target(NEXT)
        self.assertEqual(body_node.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

        else_node = while_node.target(ELSE)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[LEAVE])

    def test_while_with_continue(self):
        code = """\
def f():
    while some_condition:
        if not_interesting:
            continue
        do_something()
    else:
        do_no_break_stuff()
"""
        function_context, while_node = self._function_context(code)

        self.assertEqual(while_node.edge_names, {ELSE, NEXT, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])

        test_node = while_node.target(NEXT)
        self.assertEqual(test_node.target(RAISE), function_context[RAISE])
        self.assertEqual(test_node.edge_names, {IF, ELSE, RAISE})

        continue_node = test_node.target(IF)
        self.assertEqual(continue_node.edge_names, {CONTINUE})
        self.assertEqual(continue_node.target(CONTINUE), while_node)

        body_node = test_node.target(ELSE)
        self.assertEqual(body_node.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

        else_node = while_node.target(ELSE)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[LEAVE])

    def test_while_with_break(self):
        code = """\
def f():
    while some_condition:
        if not_interesting:
            break
        do_something()
    else:
        do_no_break_stuff()
"""
        function_context, while_node = self._function_context(code)

        self.assertEqual(while_node.edge_names, {ELSE, NEXT, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])

        test_node = while_node.target(NEXT)
        self.assertEqual(test_node.target(RAISE), function_context[RAISE])
        self.assertEqual(test_node.edge_names, {IF, ELSE, RAISE})

        break_node = test_node.target(IF)
        self.assertEqual(break_node.edge_names, {BREAK})
        self.assertEqual(break_node.target(BREAK), function_context[LEAVE])

        body_node = test_node.target(ELSE)
        self.assertEqual(body_node.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

        else_node = while_node.target(ELSE)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[LEAVE])

    def test_while_with_two_statements(self):
        code = """\
def f():
    while some_condition:
        do_something()
        do_something_else()
"""
        function_context, while_node = self._function_context(code)

        self.assertEqual(while_node.edge_names, {ELSE, NEXT, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])
        self.assertEqual(while_node.target(ELSE), function_context[LEAVE])

        body_node1 = while_node.target(NEXT)
        self.assertEqual(body_node1.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node1.target(RAISE), function_context[RAISE])

        body_node2 = body_node1.target(NEXT)
        self.assertEqual(body_node2.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node2.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node2.target(NEXT), while_node)

    def test_for_with_continue(self):
        code = """\
def f():
    for item in some_list:
        if not_interesting:
            continue
        do_something()
    else:
        do_no_break_stuff()
"""
        function_context, for_node = self._function_context(code)

        self.assertEqual(for_node.edge_names, {ELSE, NEXT, RAISE})
        self.assertEqual(for_node.target(RAISE), function_context[RAISE])

        test_node = for_node.target(NEXT)
        self.assertEqual(test_node.target(RAISE), function_context[RAISE])
        self.assertEqual(test_node.edge_names, {IF, ELSE, RAISE})

        continue_node = test_node.target(IF)
        self.assertEqual(continue_node.edge_names, {CONTINUE})
        self.assertEqual(continue_node.target(CONTINUE), for_node)

        body_node = test_node.target(ELSE)
        self.assertEqual(body_node.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node.target(NEXT), for_node)

        else_node = for_node.target(ELSE)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[LEAVE])

    def test_for_with_break(self):
        code = """\
def f():
    for item in some_list:
        if not_interesting:
            break
        do_something()
    else:
        do_no_break_stuff()
"""
        function_context, for_node = self._function_context(code)

        self.assertEqual(for_node.edge_names, {ELSE, NEXT, RAISE})
        self.assertEqual(for_node.target(RAISE), function_context[RAISE])

        test_node = for_node.target(NEXT)
        self.assertEqual(test_node.target(RAISE), function_context[RAISE])
        self.assertEqual(test_node.edge_names, {IF, ELSE, RAISE})

        break_node = test_node.target(IF)
        self.assertEqual(break_node.edge_names, {BREAK})
        self.assertEqual(break_node.target(BREAK), function_context[LEAVE])

        body_node = test_node.target(ELSE)
        self.assertEqual(body_node.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node.target(NEXT), for_node)

        else_node = for_node.target(ELSE)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[LEAVE])

    # Helper methods

    def _node_from_function(self, function_code):
        # Convert a function given as a code snippet to
        # the corresponding AST tree.
        module_node = compile(
            function_code, "test_cf", "exec", ast.PyCF_ONLY_AST
        )
        (function_node,) = module_node.body
        self.assertIsInstance(function_node, ast.FunctionDef)
        return function_node

    def _function_context(self, code):
        ast_node = self._node_from_function(code)
        function_context, enter = analyse_function(ast_node)
        self.assertEqual(
            sorted(function_context.keys()),
            [LEAVE, RAISE, RETURN, RETURN_VALUE],
        )
        self.assertEqual(function_context[LEAVE].edge_names, set())
        self.assertEqual(function_context[RAISE].edge_names, set())
        self.assertEqual(function_context[RETURN].edge_names, set())
        self.assertEqual(function_context[RETURN_VALUE].edge_names, set())

        return function_context, enter


if __name__ == "__main__":
    unittest.main()
