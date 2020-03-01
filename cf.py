"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""
import ast
import unittest


# To do: add links from CFNodes to the corresponding AST nodes.

class CFNode:
    """
    A node on the control flow graph.
    """
    def __init__(self):
        # Outward edges for possible control flow transfer.
        self._out = {}

    def add_edge(self, name, target):
        # For now, be careful about overwriting.
        if name in self._out:
            raise ValueError("An edge with that name already exists")
        self._out[name] = target

    @property
    def edge_names(self):
        # Names of outward edges.
        return sorted(self._out.keys())

    def target(self, edge_name):
        return self._out[edge_name]


def analyse(node, context):
    """
    Return analysis for a given node of an AST tree.

    Parameters
    ----------
    node : ast.Node
        Node of an AST tree.
    context : mapping from string to CFNode
        Mapping giving context in which this node is being analysed. This
        includes the nodes that control flow passes to in the event of
        a return (if within a function), raise or fallthrough (end), and
        information about any current loop context (for / while) for which
        a break or continue might make sense.

    Returns
    -------
    CFNode
        Graph node on the control flow graph.
    """


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
    # if there's a function context, may also have "return" and
    # "return_value" entries. If there's a loop context, may
    # have additional entries for use with 'break' and 'continue'.

    # It's convenient to iterate over statements in reverse, creating
    # a linked list from the last element in the list backwards.

    head = context["leave"]
    for stmt in reversed(stmts):
        if isinstance(stmt, ast.Pass):
            stmt_node = CFNode()
            stmt_node.add_edge("next", head)
        elif isinstance(stmt, (ast.Expr, ast.Assign)):
            stmt_node = CFNode()
            stmt_node.add_edge("raise", context["raise"])
            stmt_node.add_edge("next", head)
        elif isinstance(stmt, ast.Return):
            stmt_node = CFNode()
            if stmt.value is None:
                stmt_node.add_edge("return", context["return"])
            else:
                stmt_node.add_edge("raise", context["raise"])
                stmt_node.add_edge("return_value", context["return_value"])
        elif isinstance(stmt, ast.If):
            # Node where the if and else branches are merged.
            merge_node = CFNode()
            merge_node.add_edge("next", head)

            # XXX Should we instead copy the outer context and patch?
            if_context = {
                "raise": context["raise"],
                "leave": merge_node,
            }
            if "return_value" in context:
                if_context["return_value"] = context["return_value"]
            if "return" in context:
                if_context["return"] = context["return"]

            if_enter = analyse_statements(stmt.body, if_context)
            else_enter = analyse_statements(stmt.orelse, if_context)

            stmt_node = CFNode()
            stmt_node.add_edge("if_branch", if_enter)
            stmt_node.add_edge("else_branch", else_enter)
            stmt_node.add_edge("raise", context["raise"])
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
        "raise": CFNode(),
        "return_value": CFNode(),  # node for 'return <expr>'
        "return": CFNode(),  # node for plain valueless return
        "leave": CFNode(),  # node for leaving by falling off the end
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

        self.assertEqual(pass_node.edge_names, ["next"])
        self.assertIs(pass_node.target("next"), function_context["leave"])

    def test_analyse_single_expr_statement(self):
        code = """\
def f():
    do_something()
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, ["next", "raise"])
        self.assertEqual(stmt_node.target("next"), function_context["leave"])
        self.assertEqual(stmt_node.target("raise"), function_context["raise"])

    def test_analyse_assign(self):
        code = """\
def f():
    a = 123
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, ["next", "raise"])
        self.assertEqual(stmt_node.target("next"), function_context["leave"])
        self.assertEqual(stmt_node.target("raise"), function_context["raise"])

    def test_analyse_multiple_statements(self):
        code = """\
def f():
    do_something()
    do_something_else()
"""
        function_context, stmt1_node = self._function_context(code)

        self.assertEqual(stmt1_node.edge_names, ["next", "raise"])
        self.assertEqual(stmt1_node.target("raise"), function_context["raise"])

        stmt2_node = stmt1_node.target("next")
        self.assertEqual(stmt2_node.edge_names, ["next", "raise"])
        self.assertEqual(stmt2_node.target("next"), function_context["leave"])
        self.assertEqual(stmt2_node.target("raise"), function_context["raise"])

    def test_return_with_no_value(self):
        code = """\
def f():
    return
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, ["return"])
        self.assertEqual(
            stmt_node.target("return"), function_context["return"])

    def test_return_with_value(self):
        code = """\
def f():
    return None
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, ["raise", "return_value"])
        self.assertEqual(stmt_node.target("raise"), function_context["raise"])
        self.assertEqual(
            stmt_node.target("return_value"), function_context["return_value"])

    def test_if(self):
        code = """\
def f():
    if condition:
        a = 123
"""
        function_context, if_node = self._function_context(code)

        self.assertEqual(
            if_node.edge_names, ["else_branch", "if_branch", "raise"])
        self.assertEqual(
            if_node.target("raise"), function_context["raise"])

        if_branch = if_node.target("if_branch")
        self.assertEqual(if_branch.edge_names, ["next", "raise"])
        self.assertEqual(if_branch.target("raise"), function_context["raise"])

        merge_node = if_branch.target("next")
        self.assertEqual(merge_node.edge_names, ["next"])
        self.assertEqual(merge_node.target("next"), function_context["leave"])

        self.assertEqual(if_node.target("else_branch"), merge_node)

    def test_if_else(self):
        code = """\
def f():
    if condition:
        a = 123
    else:
        b = 456
"""
        function_context, if_node = self._function_context(code)

        self.assertEqual(
            if_node.edge_names, ["else_branch", "if_branch", "raise"])
        self.assertEqual(
            if_node.target("raise"), function_context["raise"])

        if_branch = if_node.target("if_branch")
        self.assertEqual(if_branch.edge_names, ["next", "raise"])
        self.assertEqual(if_branch.target("raise"), function_context["raise"])

        else_branch = if_node.target("else_branch")
        self.assertEqual(else_branch.edge_names, ["next", "raise"])
        self.assertEqual(else_branch.target("raise"), function_context["raise"])

        merge_node = if_branch.target("next")
        self.assertEqual(merge_node.edge_names, ["next"])
        self.assertEqual(merge_node.target("next"), function_context["leave"])

        self.assertEqual(else_branch.target("next"), merge_node)

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
            if_node.target("if_branch").edge_names,
            ["raise", "return_value"],
        )
        self.assertEqual(
            if_node.target("if_branch").target("return_value"),
            function_context["return_value"],
        )
        self.assertEqual(
            if_node.target("if_branch").target("raise"),
            function_context["raise"],
        )

        self.assertEqual(
            if_node.target("else_branch").edge_names,
            ["raise", "return_value"],
        )
        self.assertEqual(
            if_node.target("else_branch").target("return_value"),
            function_context["return_value"],
        )
        self.assertEqual(
            if_node.target("else_branch").target("raise"),
            function_context["raise"],
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
            if_node.target("if_branch").edge_names,
            ["return"],
        )
        self.assertEqual(
            if_node.target("if_branch").target("return"),
            function_context["return"],
        )

        self.assertEqual(
            if_node.target("else_branch").edge_names,
            ["return"],
        )
        self.assertEqual(
            if_node.target("else_branch").target("return"),
            function_context["return"],
        )

    def test_unreachable_statements(self):
        code = """\
def f():
    do_something()
    return
    do_something_else()
"""
        function_context, stmt1_node = self._function_context(code)

        self.assertEqual(stmt1_node.edge_names, ["next", "raise"])
        self.assertEqual(stmt1_node.target("raise"), function_context["raise"])

        stmt2_node = stmt1_node.target("next")
        self.assertEqual(stmt2_node.edge_names, ["return"])
        self.assertEqual(
            stmt2_node.target("return"), function_context["return"])

    # Helper methods

    def _node_from_function(self, function_code):
        # Convert a function given as a code snippet to
        # the corresponding AST tree.
        module_node = compile(
            function_code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        function_node, = module_node.body
        self.assertIsInstance(function_node, ast.FunctionDef)
        return function_node

    def _function_context(self, code):
        ast_node = self._node_from_function(code)
        function_context, enter = analyse_function(ast_node)
        self.assertEqual(
            sorted(function_context.keys()),
            ["leave", "raise", "return", "return_value"],
        )
        self.assertEqual(function_context["leave"].edge_names, [])
        self.assertEqual(function_context["raise"].edge_names, [])
        self.assertEqual(function_context["return"].edge_names, [])
        self.assertEqual(function_context["return_value"].edge_names, [])
        # self.assertEqual(function_context["enter"].edge_names, ["next"])

        return function_context, enter


if __name__ == "__main__":
    unittest.main()
