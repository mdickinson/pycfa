"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""

# TODO: add links from CFNodes to the corresponding AST nodes.
# TODO: CFGraph object, containing both nodes and edges. (This
#       will allow easier detection of unreachable statements.)
# TODO: Coverage for other statement types.
# TODO: Better context management (more functional).
# TODO: In tests, remove function context from code snippets where it's not
#       needed.
# TODO: create finally clauses lazily, instead of always creating
#       all six.


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
        context, pass_node = self._function_context(code)

        self.assertEdges(pass_node, {NEXT})
        self.assertEqual(pass_node.target(NEXT), context[NEXT])

    def test_analyse_single_expr_statement(self):
        code = """\
def f():
    do_something()
"""
        context, stmt_node = self._function_context(code)

        self.assertEdges(stmt_node, {NEXT, RAISE})
        self.assertEqual(stmt_node.target(NEXT), context[NEXT])
        self.assertEqual(stmt_node.target(RAISE), context[RAISE])

    def test_analyse_assign(self):
        code = """\
def f():
    a = 123
"""
        context, stmt_node = self._function_context(code)

        self.assertEdges(stmt_node, {NEXT, RAISE})
        self.assertEqual(stmt_node.target(NEXT), context[NEXT])
        self.assertEqual(stmt_node.target(RAISE), context[RAISE])

    def test_analyse_multiple_statements(self):
        code = """\
def f():
    do_something()
    do_something_else()
"""
        context, stmt1_node = self._function_context(code)

        self.assertEdges(stmt1_node, {NEXT, RAISE})
        self.assertEqual(stmt1_node.target(RAISE), context[RAISE])

        stmt2_node = stmt1_node.target(NEXT)
        self.assertEdges(stmt2_node, {NEXT, RAISE})
        self.assertEqual(stmt2_node.target(NEXT), context[NEXT])
        self.assertEqual(stmt2_node.target(RAISE), context[RAISE])

    def test_return_with_no_value(self):
        code = """\
def f():
    return
"""
        context, stmt_node = self._function_context(code)

        self.assertEdges(stmt_node, {RETURN})
        self.assertEqual(stmt_node.target(RETURN), context[RETURN])

    def test_return_with_value(self):
        code = """\
def f():
    return None
"""
        context, stmt_node = self._function_context(code)

        self.assertEdges(stmt_node, {RAISE, RETURN_VALUE})
        self.assertEqual(stmt_node.target(RAISE), context[RAISE])
        self.assertEqual(stmt_node.target(RETURN_VALUE), context[RETURN_VALUE])

    def test_raise(self):
        code = """\
def f():
    raise TypeError("don't call me")
"""
        context, stmt_node = self._function_context(code)

        self.assertEdges(stmt_node, {RAISE})
        self.assertEqual(stmt_node.target(RAISE), context[RAISE])

    def test_if(self):
        code = """\
def f():
    if condition:
        a = 123
"""
        context, if_node = self._function_context(code)

        self.assertEdges(if_node, {ELSE, IF, RAISE})
        self.assertEqual(if_node.target(RAISE), context[RAISE])

        if_branch = if_node.target(IF)
        self.assertEdges(if_branch, {NEXT, RAISE})
        self.assertEqual(if_branch.target(RAISE), context[RAISE])
        self.assertEqual(if_branch.target(NEXT), context[NEXT])
        self.assertEqual(if_node.target(ELSE), context[NEXT])

    def test_if_else(self):
        code = """\
def f():
    if condition:
        a = 123
    else:
        b = 456
"""
        context, if_node = self._function_context(code)

        self.assertEdges(if_node, {ELSE, IF, RAISE})
        self.assertEqual(if_node.target(RAISE), context[RAISE])

        if_branch = if_node.target(IF)
        self.assertEdges(if_branch, {NEXT, RAISE})
        self.assertEqual(if_branch.target(RAISE), context[RAISE])
        self.assertEqual(if_branch.target(NEXT), context[NEXT])

        else_branch = if_node.target(ELSE)
        self.assertEdges(else_branch, {NEXT, RAISE})
        self.assertEqual(else_branch.target(RAISE), context[RAISE])
        self.assertEqual(else_branch.target(NEXT), context[NEXT])

    def test_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return 123
    else:
        return 456
"""
        context, if_node = self._function_context(code)
        if_branch = if_node.target(IF)
        self.assertEdges(if_branch, {RAISE, RETURN_VALUE})
        self.assertEqual(if_branch.target(RETURN_VALUE), context[RETURN_VALUE])
        self.assertEqual(if_branch.target(RAISE), context[RAISE])

        else_node = if_node.target(ELSE)
        self.assertEdges(else_node, {RAISE, RETURN_VALUE})
        self.assertEqual(else_node.target(RETURN_VALUE), context[RETURN_VALUE])
        self.assertEqual(else_node.target(RAISE), context[RAISE])

    def test_plain_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return
    else:
        return
"""
        context, if_node = self._function_context(code)
        if_branch = if_node.target(IF)
        self.assertEdges(if_branch, {RETURN})
        self.assertEqual(if_branch.target(RETURN), context[RETURN])

        else_node = if_node.target(ELSE)
        self.assertEdges(else_node, {RETURN})
        self.assertEqual(else_node.target(RETURN), context[RETURN])

    def test_unreachable_statements(self):
        code = """\
def f():
    do_something()
    return
    do_something_else()
"""
        context, stmt1_node = self._function_context(code)

        self.assertEdges(stmt1_node, {NEXT, RAISE})
        self.assertEqual(stmt1_node.target(RAISE), context[RAISE])

        stmt2_node = stmt1_node.target(NEXT)
        self.assertEdges(stmt2_node, {RETURN})
        self.assertEqual(stmt2_node.target(RETURN), context[RETURN])

    def test_while(self):
        code = """\
def f():
    while some_condition:
        do_something()
"""
        context, while_node = self._function_context(code)

        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), context[RAISE])
        self.assertEqual(while_node.target(ELSE), context[NEXT])

        body_node = while_node.target(ENTER)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

    def test_while_else(self):
        code = """\
def f():
    while some_condition:
        do_something()
    else:
        do_no_break_stuff()
"""
        context, while_node = self._function_context(code)

        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), context[RAISE])

        body_node = while_node.target(ENTER)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

        else_node = while_node.target(ELSE)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), context[RAISE])
        self.assertEqual(else_node.target(NEXT), context[NEXT])

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
        context, while_node = self._function_context(code)

        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), context[RAISE])

        test_node = while_node.target(ENTER)
        self.assertEqual(test_node.target(RAISE), context[RAISE])
        self.assertEdges(test_node, {IF, ELSE, RAISE})

        continue_node = test_node.target(IF)
        self.assertEdges(continue_node, {CONTINUE})
        self.assertEqual(continue_node.target(CONTINUE), while_node)

        body_node = test_node.target(ELSE)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

        else_node = while_node.target(ELSE)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), context[RAISE])
        self.assertEqual(else_node.target(NEXT), context[NEXT])

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
        context, while_node = self._function_context(code)

        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), context[RAISE])

        test_node = while_node.target(ENTER)
        self.assertEqual(test_node.target(RAISE), context[RAISE])
        self.assertEdges(test_node, {IF, ELSE, RAISE})

        break_node = test_node.target(IF)
        self.assertEdges(break_node, {BREAK})
        self.assertEqual(break_node.target(BREAK), context[NEXT])

        body_node = test_node.target(ELSE)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), context[RAISE])
        self.assertEqual(body_node.target(NEXT), while_node)

        else_node = while_node.target(ELSE)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), context[RAISE])
        self.assertEqual(else_node.target(NEXT), context[NEXT])

    def test_while_with_two_statements(self):
        code = """\
def f():
    while some_condition:
        do_something()
        do_something_else()
"""
        context, while_node = self._function_context(code)

        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEqual(while_node.target(RAISE), context[RAISE])
        self.assertEqual(while_node.target(ELSE), context[NEXT])

        body_node1 = while_node.target(ENTER)
        self.assertEdges(body_node1, {NEXT, RAISE})
        self.assertEqual(body_node1.target(RAISE), context[RAISE])

        body_node2 = body_node1.target(NEXT)
        self.assertEdges(body_node2, {NEXT, RAISE})
        self.assertEqual(body_node2.target(RAISE), context[RAISE])
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
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        test_node = for_node.target(ENTER)
        self.assertEqual(test_node.target(RAISE), context[RAISE])
        self.assertEdges(test_node, {IF, ELSE, RAISE})

        continue_node = test_node.target(IF)
        self.assertEdges(continue_node, {CONTINUE})
        self.assertEqual(continue_node.target(CONTINUE), for_node)

        body_node = test_node.target(ELSE)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), context[RAISE])
        self.assertEqual(body_node.target(NEXT), for_node)

        else_node = for_node.target(ELSE)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), context[RAISE])
        self.assertEqual(else_node.target(NEXT), context[NEXT])

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
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        test_node = for_node.target(ENTER)
        self.assertEqual(test_node.target(RAISE), context[RAISE])
        self.assertEdges(test_node, {IF, ELSE, RAISE})

        break_node = test_node.target(IF)
        self.assertEdges(break_node, {BREAK})
        self.assertEqual(break_node.target(BREAK), context[NEXT])

        body_node = test_node.target(ELSE)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), context[RAISE])
        self.assertEqual(body_node.target(NEXT), for_node)

        else_node = for_node.target(ELSE)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), context[RAISE])
        self.assertEqual(else_node.target(NEXT), context[NEXT])

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
        context, try_node = self._function_context(code)

        # starting node is the something_dangerous node; we don't
        # have a separate node for the "try" itself.
        self.assertEdges(try_node, {NEXT, RAISE})

        except1_node = try_node.target(RAISE)
        self.assertEdges(except1_node, {MATCH, NO_MATCH, RAISE})
        self.assertEqual(except1_node.target(RAISE), context[RAISE])

        match1_node = except1_node.target(MATCH)
        self.assertEdges(match1_node, {NEXT, RAISE})
        self.assertEqual(match1_node.target(RAISE), context[RAISE])
        self.assertEqual(match1_node.target(NEXT), context[NEXT])

        match2_node = except1_node.target(NO_MATCH)
        self.assertEdges(match2_node, {NEXT, RAISE})
        self.assertEqual(match2_node.target(RAISE), context[RAISE])
        self.assertEqual(match2_node.target(NEXT), context[NEXT])

        else_node = try_node.target(NEXT)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEqual(else_node.target(RAISE), context[RAISE])
        self.assertEqual(else_node.target(NEXT), context[NEXT])

    def test_try_except_pass(self):
        code = """\
def f():
    try:
        something_dangerous()
    except:
        pass
"""
        context, try_node = self._function_context(code)

        # In this case, context[RAISE] is not reachable.
        self.assertEdges(try_node, {NEXT, RAISE})
        self.assertEqual(try_node.target(NEXT), context[NEXT])

        pass_node = try_node.target(RAISE)
        self.assertEdges(pass_node, {NEXT})
        self.assertEqual(pass_node.target(NEXT), context[NEXT])

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
        context, try_node = self._function_context(code)

        self.assertEdges(try_node, {RAISE})

        except_node = try_node.target(RAISE)
        self.assertEdges(except_node, {MATCH, NO_MATCH, RAISE})
        self.assertEqual(except_node.target(NO_MATCH), context[RAISE])
        self.assertEqual(except_node.target(RAISE), context[RAISE])

        pass_node = except_node.target(MATCH)
        self.assertEdges(pass_node, {NEXT})
        self.assertEqual(pass_node.target(NEXT), context[NEXT])

    def test_try_finally_pass(self):
        code = """\
def f():
    try:
        pass
    finally:
        do_something()
"""
        context, try_node = self._function_context(code)

        self.assertEdges(try_node, {NEXT})

        finally_node = try_node.target(NEXT)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_node.target(NEXT), context[NEXT])

    def test_try_finally_raise(self):
        code = """\
def f():
    try:
        raise ValueError()
    finally:
        do_something()
"""
        context, try_node = self._function_context(code)

        self.assertEdges(try_node, {RAISE})

        finally_node = try_node.target(RAISE)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_node.target(NEXT), context[RAISE])

    def test_try_finally_return(self):
        code = """\
def f():
    try:
        return
    finally:
        do_something()
"""
        context, try_node = self._function_context(code)

        self.assertEdges(try_node, {RETURN})

        finally_node = try_node.target(RETURN)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_node.target(NEXT), context[RETURN])

    def test_try_finally_return_value(self):
        code = """\
def f():
    try:
        return "abc"
    finally:
        do_something()
"""
        context, try_node = self._function_context(code)

        self.assertEdges(try_node, {RAISE, RETURN_VALUE})

        finally_node = try_node.target(RETURN_VALUE)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_node.target(NEXT), context[RETURN_VALUE])

        finally2_node = try_node.target(RAISE)
        self.assertEdges(finally2_node, {NEXT, RAISE})
        self.assertEqual(finally2_node.target(RAISE), context[RAISE])
        self.assertEqual(finally2_node.target(NEXT), context[RAISE])

    def test_try_finally_break(self):
        code = """\
def f():
    for item in items:
        try:
            break
        finally:
            do_something()
"""
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), context[NEXT])
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        try_node = for_node.target(ENTER)
        self.assertEdges(try_node, {BREAK})

        finally_node = try_node.target(BREAK)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_node.target(NEXT), context[NEXT])

    def test_try_finally_continue(self):
        code = """\
def f():
    for item in items:
        try:
            continue
        finally:
            do_something()
"""
        context, for_node = self._function_context(code)

        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEqual(for_node.target(ELSE), context[NEXT])
        self.assertEqual(for_node.target(RAISE), context[RAISE])

        try_node = for_node.target(ENTER)
        self.assertEdges(try_node, {CONTINUE})

        finally_node = try_node.target(CONTINUE)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEqual(finally_node.target(RAISE), context[RAISE])
        self.assertEqual(finally_node.target(NEXT), for_node)

    def test_return_value_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return some_value()
"""
        context, raise_node = self._function_context(code)

        self.assertEdges(raise_node, {RAISE})

        return_node = raise_node.target(RAISE)
        self.assertEdges(return_node, {RAISE, RETURN_VALUE})
        self.assertEqual(
            return_node.target(RETURN_VALUE), context[RETURN_VALUE]
        )
        self.assertEqual(return_node.target(RAISE), context[RAISE])

    def test_return_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return
"""
        context, raise_node = self._function_context(code)

        self.assertEdges(raise_node, {RAISE})

        return_node = raise_node.target(RAISE)
        self.assertEdges(return_node, {RETURN})
        self.assertEqual(return_node.target(RETURN), context[RETURN])

    def test_raise_in_finally(self):
        code = """\
def f():
    try:
        pass
    finally:
        raise SomeException()
"""
        context, pass_node = self._function_context(code)

        self.assertEdges(pass_node, {NEXT})

        raise_node = pass_node.target(NEXT)
        self.assertEdges(raise_node, {RAISE})
        self.assertEqual(raise_node.target(RAISE), context[RAISE])

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

        continue_node = raise_node.target(RAISE)
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

        continue_node = raise_node.target(RAISE)
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

        pass_node = try_node.target(RAISE)
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

        self.assertEdges(assign_node, {NEXT, RAISE})
        self.assertEqual(assign_node.target(RAISE), context[RAISE])

        try_node = assign_node.target(NEXT)
        self.assertEdges(try_node, {NEXT, RAISE})
        self.assertEqual(try_node.target(NEXT), context[NEXT])

        pass_node = try_node.target(RAISE)
        self.assertEdges(pass_node, {NEXT})
        self.assertEqual(pass_node.target(NEXT), context[NEXT])

    def test_with(self):
        code = """\
with some_cm() as name:
    do_something()
"""
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        context = {
            NEXT: CFNode(),
            RAISE: CFNode(),
        }
        with_node = analyse_statements(module_node.body, context)
        self.assertEdges(with_node, {ENTER, RAISE})
        self.assertEqual(with_node.target(RAISE), context[RAISE])

        body_node = with_node.target(ENTER)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEqual(body_node.target(RAISE), context[RAISE])
        self.assertEqual(body_node.target(NEXT), context[NEXT])

    def test_global(self):
        code = """\
def f():
    global bob
"""
        context, node = self._function_context(code)
        self.assertEqual(node, context[NEXT])

    def test_nonlocal(self):
        code = """\
def f(bob):
    def g():
        nonlocal bob
"""
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        (function_node,) = module_node.body
        (inner_function,) = function_node.body

        context, node = analyse_function(inner_function)
        self.assertEqual(node, context[NEXT])

    def test_assorted_simple_statements(self):
        code = """\
del x, y, z
def f():
    pass
from france import cheese
import this
a += b
class A:
    pass
"""
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        context = {
            NEXT: CFNode(),
            RAISE: CFNode(),
        }

        node = analyse_statements(module_node.body, context)
        for _ in range(len(module_node.body)):
            self.assertEdges(node, {NEXT, RAISE})
            self.assertEqual(node.target(RAISE), context[RAISE])
            node = node.target(NEXT)

        self.assertEqual(node, context[NEXT])

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
        context, enter = analyse_function(ast_node)
        self.assertEqual(
            sorted(context.keys()), [NEXT, RAISE, RETURN, RETURN_VALUE],
        )
        self.assertEdges(context[NEXT], set())
        self.assertEdges(context[RAISE], set())
        self.assertEdges(context[RETURN], set())
        self.assertEdges(context[RETURN_VALUE], set())

        return context, enter


if __name__ == "__main__":
    unittest.main()
