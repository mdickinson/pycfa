"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""
# TODO: add links from CFNodes to the corresponding AST nodes.
# TODO: try/finally


import ast
import unittest

from cf import (
    analyse_function,
    BREAK,
    CONTINUE,
    ELSE,
    IF,
    LEAVE,
    MATCH,
    NO_MATCH,
    NEXT,
    RAISE,
    RETURN,
    RETURN_VALUE,
)


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

    def test_try_except_else(self):
        code = """\
def f():
    try:
        something_dangerous()
    except ValueError:
        do_something()
    except:
        do_something_else()
    else:
        all_okay()
"""
        function_context, try_node = self._function_context(code)

        # starting node is the something_dangerous node; we don't
        # have a separate node for the "try" itself.
        self.assertEqual(try_node.edge_names, {NEXT, RAISE})

        except1_node = try_node.target(RAISE)
        self.assertEqual(except1_node.edge_names, {MATCH, NO_MATCH, RAISE})
        self.assertEqual(except1_node.target(RAISE), function_context[RAISE])

        match1_node = except1_node.target(MATCH)
        self.assertEqual(match1_node.edge_names, {NEXT, RAISE})
        self.assertEqual(match1_node.target(RAISE), function_context[RAISE])
        self.assertEqual(match1_node.target(NEXT), function_context[LEAVE])

        except2_node = except1_node.target(NO_MATCH)
        self.assertEqual(except2_node.edge_names, {MATCH})

        match2_node = except2_node.target(MATCH)
        self.assertEqual(match2_node.edge_names, {NEXT, RAISE})
        self.assertEqual(match2_node.target(RAISE), function_context[RAISE])
        self.assertEqual(match2_node.target(NEXT), function_context[LEAVE])

        else_node = try_node.target(NEXT)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[LEAVE])

    def test_try_except_pass(self):
        code = """\
def f():
    try:
        something_dangerous()
    except:
        pass
"""
        function_context, try_node = self._function_context(code)

        # In this case, function_context[RAISE] is not reachable.
        self.assertEqual(try_node.edge_names, {NEXT, RAISE})
        self.assertEqual(try_node.target(NEXT), function_context[LEAVE])

        except_node = try_node.target(RAISE)
        self.assertEqual(except_node.edge_names, {MATCH})

        pass_node = except_node.target(MATCH)
        self.assertEqual(pass_node.edge_names, {NEXT})
        self.assertEqual(pass_node.target(NEXT), function_context[LEAVE])

    def test_raise_in_try(self):
        code = """\
def f():
    try:
        raise ValueError()
    except SomeException:
        pass
    else:
        # unreachable
        return
"""
        function_context, try_node = self._function_context(code)

        self.assertEqual(try_node.edge_names, {RAISE})

        except_node = try_node.target(RAISE)
        self.assertEqual(except_node.edge_names, {MATCH, NO_MATCH, RAISE})
        self.assertEqual(except_node.target(NO_MATCH), function_context[RAISE])
        self.assertEqual(except_node.target(RAISE), function_context[RAISE])

        pass_node = except_node.target(MATCH)
        self.assertEqual(pass_node.edge_names, {NEXT})
        self.assertEqual(pass_node.target(NEXT), function_context[LEAVE])

    def test_try_finally_pass(self):
        code = """\
def f():
    try:
        pass
    finally:
        do_something()
"""
        function_context, try_node = self._function_context(code)

        self.assertEqual(try_node.edge_names, {NEXT})

        finally_node = try_node.target(NEXT)
        self.assertEqual(finally_node.edge_names, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), function_context[RAISE])
        self.assertEqual(finally_node.target(NEXT), function_context[LEAVE])

    def test_try_finally_raise(self):
        code = """\
def f():
    try:
        raise ValueError()
    finally:
        do_something()
"""
        function_context, try_node = self._function_context(code)

        self.assertEqual(try_node.edge_names, {RAISE})

        finally_node = try_node.target(RAISE)
        self.assertEqual(finally_node.edge_names, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), function_context[RAISE])
        self.assertEqual(finally_node.target(NEXT), function_context[RAISE])

    def test_try_finally_return(self):
        code = """\
def f():
    try:
        return
    finally:
        do_something()
"""
        function_context, try_node = self._function_context(code)

        self.assertEqual(try_node.edge_names, {RETURN})

        finally_node = try_node.target(RETURN)
        self.assertEqual(finally_node.edge_names, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), function_context[RAISE])
        self.assertEqual(finally_node.target(NEXT), function_context[RETURN])

    def test_try_finally_return_value(self):
        code = """\
def f():
    try:
        return "abc"
    finally:
        do_something()
"""
        function_context, try_node = self._function_context(code)

        self.assertEqual(try_node.edge_names, {RAISE, RETURN_VALUE})

        finally_node = try_node.target(RETURN_VALUE)
        self.assertEqual(finally_node.edge_names, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), function_context[RAISE])
        self.assertEqual(
            finally_node.target(NEXT), function_context[RETURN_VALUE])

        finally2_node = try_node.target(RAISE)
        self.assertEqual(finally2_node.edge_names, {NEXT, RAISE})
        self.assertEqual(finally2_node.target(RAISE), function_context[RAISE])
        self.assertEqual(
            finally2_node.target(NEXT), function_context[RAISE])

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
