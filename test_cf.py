"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""

# TODO: add and use assertEdges method.
# TODO: use context in place of function_context.
# TODO: add links from CFNodes to the corresponding AST nodes.
# TODO: try/finally
# TODO: CFGraph object, containing both nodes and edges. (This
#       will allow easier detection of unreachable statements.)
# TODO: Better context management (more functional).
# TODO: Remove function context from code snippets where it's not needed.
# TODO: get rid of bare except clause as a node?


import ast
import unittest

from cf import (
    analyse_function,
    analyse_statements,
    BREAK,
    CFNode,
    CONTINUE,
    ELSE,
    ENTER,
    IF,
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
        self.assertIs(pass_node.target(NEXT), function_context[NEXT])

    def test_analyse_single_expr_statement(self):
        code = """\
def f():
    do_something()
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, {NEXT, RAISE})
        self.assertEqual(stmt_node.target(NEXT), function_context[NEXT])
        self.assertEqual(stmt_node.target(RAISE), function_context[RAISE])

    def test_analyse_assign(self):
        code = """\
def f():
    a = 123
"""
        function_context, stmt_node = self._function_context(code)

        self.assertEqual(stmt_node.edge_names, {NEXT, RAISE})
        self.assertEqual(stmt_node.target(NEXT), function_context[NEXT])
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
        self.assertEqual(stmt2_node.target(NEXT), function_context[NEXT])
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
        self.assertEqual(if_branch.target(NEXT), function_context[NEXT])
        self.assertEqual(if_node.target(ELSE), function_context[NEXT])

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
        self.assertEqual(if_branch.target(NEXT), function_context[NEXT])

        else_branch = if_node.target(ELSE)
        self.assertEqual(else_branch.edge_names, {NEXT, RAISE})
        self.assertEqual(else_branch.target(RAISE), function_context[RAISE])
        self.assertEqual(else_branch.target(NEXT), function_context[NEXT])

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

        self.assertEqual(while_node.edge_names, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])
        self.assertEqual(while_node.target(ELSE), function_context[NEXT])

        body_node = while_node.target(ENTER)
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

        self.assertEqual(while_node.edge_names, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])

        body_node = while_node.target(ENTER)
        self.assertEqual(body_node.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

        else_node = while_node.target(ELSE)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[NEXT])

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

        self.assertEqual(while_node.edge_names, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])

        test_node = while_node.target(ENTER)
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
        self.assertEqual(else_node.target(NEXT), function_context[NEXT])

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

        self.assertEqual(while_node.edge_names, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])

        test_node = while_node.target(ENTER)
        self.assertEqual(test_node.target(RAISE), function_context[RAISE])
        self.assertEqual(test_node.edge_names, {IF, ELSE, RAISE})

        break_node = test_node.target(IF)
        self.assertEqual(break_node.edge_names, {BREAK})
        self.assertEqual(break_node.target(BREAK), function_context[NEXT])

        body_node = test_node.target(ELSE)
        self.assertEqual(body_node.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

        else_node = while_node.target(ELSE)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[NEXT])

    def test_while_with_two_statements(self):
        code = """\
def f():
    while some_condition:
        do_something()
        do_something_else()
"""
        function_context, while_node = self._function_context(code)

        self.assertEqual(while_node.edge_names, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), function_context[RAISE])
        self.assertEqual(while_node.target(ELSE), function_context[NEXT])

        body_node1 = while_node.target(ENTER)
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

        self.assertEqual(for_node.edge_names, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(RAISE), function_context[RAISE])

        test_node = for_node.target(ENTER)
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
        self.assertEqual(else_node.target(NEXT), function_context[NEXT])

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

        self.assertEqual(for_node.edge_names, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(RAISE), function_context[RAISE])

        test_node = for_node.target(ENTER)
        self.assertEqual(test_node.target(RAISE), function_context[RAISE])
        self.assertEqual(test_node.edge_names, {IF, ELSE, RAISE})

        break_node = test_node.target(IF)
        self.assertEqual(break_node.edge_names, {BREAK})
        self.assertEqual(break_node.target(BREAK), function_context[NEXT])

        body_node = test_node.target(ELSE)
        self.assertEqual(body_node.edge_names, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), function_context[RAISE])
        self.assertEqual(body_node.target(NEXT), for_node)

        else_node = for_node.target(ELSE)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[NEXT])

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
        self.assertEqual(match1_node.target(NEXT), function_context[NEXT])

        except2_node = except1_node.target(NO_MATCH)
        self.assertEqual(except2_node.edge_names, {MATCH})

        match2_node = except2_node.target(MATCH)
        self.assertEqual(match2_node.edge_names, {NEXT, RAISE})
        self.assertEqual(match2_node.target(RAISE), function_context[RAISE])
        self.assertEqual(match2_node.target(NEXT), function_context[NEXT])

        else_node = try_node.target(NEXT)
        self.assertEqual(else_node.edge_names, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), function_context[RAISE])
        self.assertEqual(else_node.target(NEXT), function_context[NEXT])

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
        self.assertEqual(try_node.target(NEXT), function_context[NEXT])

        except_node = try_node.target(RAISE)
        self.assertEqual(except_node.edge_names, {MATCH})

        pass_node = except_node.target(MATCH)
        self.assertEqual(pass_node.edge_names, {NEXT})
        self.assertEqual(pass_node.target(NEXT), function_context[NEXT])

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
        self.assertEqual(pass_node.target(NEXT), function_context[NEXT])

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
        self.assertEqual(finally_node.target(NEXT), function_context[NEXT])

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
            finally_node.target(NEXT), function_context[RETURN_VALUE]
        )

        finally2_node = try_node.target(RAISE)
        self.assertEqual(finally2_node.edge_names, {NEXT, RAISE})
        self.assertEqual(finally2_node.target(RAISE), function_context[RAISE])
        self.assertEqual(finally2_node.target(NEXT), function_context[RAISE])

    def test_try_finally_break(self):
        code = """\
def f():
    for item in items:
        try:
            break
        finally:
            do_something()
"""
        function_context, for_node = self._function_context(code)

        self.assertEqual(for_node.edge_names, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), function_context[NEXT])
        self.assertEqual(for_node.target(RAISE), function_context[RAISE])

        try_node = for_node.target(ENTER)
        self.assertEqual(try_node.edge_names, {BREAK})

        finally_node = try_node.target(BREAK)
        self.assertEqual(finally_node.edge_names, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), function_context[RAISE])
        self.assertEqual(finally_node.target(NEXT), function_context[NEXT])

    def test_try_finally_continue(self):
        code = """\
def f():
    for item in items:
        try:
            continue
        finally:
            do_something()
"""
        function_context, for_node = self._function_context(code)

        self.assertEqual(for_node.edge_names, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), function_context[NEXT])
        self.assertEqual(for_node.target(RAISE), function_context[RAISE])

        try_node = for_node.target(ENTER)
        self.assertEqual(try_node.edge_names, {CONTINUE})

        finally_node = try_node.target(CONTINUE)
        self.assertEqual(finally_node.edge_names, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), function_context[RAISE])
        self.assertEqual(finally_node.target(NEXT), for_node)

    def test_return_value_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return some_value()
"""
        function_context, raise_node = self._function_context(code)

        self.assertEqual(raise_node.edge_names, {RAISE})

        return_node = raise_node.target(RAISE)
        self.assertEqual(return_node.edge_names, {RAISE, RETURN_VALUE})
        self.assertEqual(
            return_node.target(RETURN_VALUE), function_context[RETURN_VALUE]
        )
        self.assertEqual(return_node.target(RAISE), function_context[RAISE])

    def test_return_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return
"""
        function_context, raise_node = self._function_context(code)

        self.assertEqual(raise_node.edge_names, {RAISE})

        return_node = raise_node.target(RAISE)
        self.assertEqual(return_node.edge_names, {RETURN})
        self.assertEqual(return_node.target(RETURN), function_context[RETURN])

    def test_raise_in_finally(self):
        code = """\
def f():
    try:
        pass
    finally:
        raise SomeException()
"""
        function_context, pass_node = self._function_context(code)

        self.assertEqual(pass_node.edge_names, {NEXT})

        raise_node = pass_node.target(NEXT)
        self.assertEqual(raise_node.edge_names, {RAISE})
        self.assertEqual(raise_node.target(RAISE), function_context[RAISE])

    def test_break_in_finally(self):
        code = """\
def f():
    for item in item_factory():
        try:
            # will be superseded by the break in the finally
            return
        finally:
            break
"""
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), context[NEXT])
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        return_node = for_node.target(ENTER)
        self.assertEdges(return_node, {RETURN})

        break_node = return_node.target(RETURN)
        self.assertEdges(break_node, {BREAK})
        self.assertEqual(break_node.target(BREAK), context[NEXT])

    def test_continue_in_finally(self):
        code = """\
def f():
    for item in item_factory():
        try:
            # will be superseded by the continue in the finally
            raise SomeException()
        finally:
            continue
"""
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), context[NEXT])
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        raise_node = for_node.target(ENTER)
        self.assertEdges(raise_node, {RAISE})

        continue_node = raise_node.target(RAISE)
        self.assertEdges(continue_node, {CONTINUE})
        self.assertEqual(continue_node.target(CONTINUE), for_node)

    def test_continue_in_except_no_finally(self):
        code = """\
def f():
    for item in item_factory():
        try:
            raise SomeException()
        except:
            continue
"""
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), context[NEXT])
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        raise_node = for_node.target(ENTER)
        self.assertEdges(raise_node, {RAISE})

        except_node = raise_node.target(RAISE)
        self.assertEdges(except_node, {MATCH})

        continue_node = except_node.target(MATCH)
        self.assertEdges(continue_node, {CONTINUE})
        self.assertEqual(continue_node.target(CONTINUE), for_node)

    def test_continue_in_except(self):
        code = """\
def f():
    for item in item_factory():
        try:
            raise SomeException()
        except:
            continue
        finally:
            do_cleanup()
"""
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), context[NEXT])
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        raise_node = for_node.target(ENTER)
        self.assertEdges(raise_node, {RAISE})

        except_node = raise_node.target(RAISE)
        self.assertEdges(except_node, {MATCH})

        continue_node = except_node.target(MATCH)
        self.assertEdges(continue_node, {CONTINUE})

        finally_node = continue_node.target(CONTINUE)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_node.target(NEXT), for_node)

    def test_break_in_except(self):
        code = """\
def f():
    for item in item_factory():
        try:
            raise SomeException()
        except OtherException():
            break
        finally:
            do_cleanup()
"""
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), context[NEXT])
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        raise_node = for_node.target(ENTER)
        self.assertEdges(raise_node, {RAISE})

        except_node = raise_node.target(RAISE)
        self.assertEdges(except_node, {MATCH, NO_MATCH, RAISE})
        self.assertEqual(
            except_node.target(RAISE), except_node.target(NO_MATCH)
        )

        finally_raise_node = except_node.target(RAISE)
        self.assertEdges(finally_raise_node, {NEXT, RAISE})
        self.assertEqual(finally_raise_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_raise_node.target(NEXT), context[RAISE])

        break_node = except_node.target(MATCH)
        self.assertEdges(break_node, {BREAK})

        finally_node = break_node.target(BREAK)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_node.target(NEXT), context[NEXT])

    def test_return_in_try_else(self):
        code = """\
def f():
    for item in item_factory():
        try:
            do_something()
        except:
            pass
        else:
            return
        finally:
            do_cleanup()
"""
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), context[NEXT])
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        try_node = for_node.target(ENTER)
        self.assertEdges(try_node, {NEXT, RAISE})

        except_node = try_node.target(RAISE)
        self.assertEdges(except_node, {MATCH})

        pass_node = except_node.target(MATCH)
        self.assertEdges(pass_node, {NEXT})

        finally1_node = pass_node.target(NEXT)
        self.assertEdges(finally1_node, {NEXT, RAISE})
        self.assertEqual(finally1_node.target(RAISE), context[RAISE])
        self.assertEqual(finally1_node.target(NEXT), for_node)

        else_node = try_node.target(NEXT)
        self.assertEdges(else_node, {RETURN})

        finally_node = else_node.target(RETURN)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_node.target(NEXT), context[RETURN])

    def test_statements_outside_function(self):
        code = """\
a = calculate()
try:
    something()
except:
    pass
"""
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        context = {
            NEXT: CFNode(),
            RAISE: CFNode(),
        }
        assign_node = analyse_statements(module_node.body, context)

        self.assertEqual(assign_node.edge_names, {NEXT, RAISE})
        self.assertEqual(assign_node.target(RAISE), context[RAISE])

    # Assertions

    def assertEdges(self, node, edges):
        """
        Assert that the outward edges from a node have the given names.
        """
        self.assertEqual(node.edge_names, edges)

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
            [NEXT, RAISE, RETURN, RETURN_VALUE],
        )
        self.assertEqual(function_context[NEXT].edge_names, set())
        self.assertEqual(function_context[RAISE].edge_names, set())
        self.assertEqual(function_context[RETURN].edge_names, set())
        self.assertEqual(function_context[RETURN_VALUE].edge_names, set())

        return function_context, enter


if __name__ == "__main__":
    unittest.main()
