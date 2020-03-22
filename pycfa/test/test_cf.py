"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""

# TODO: move edge labels to cfgraph?
# TODO: add node types, and document edge labels for each node type?
# TODO: make break/continue have only NEXT edge labels?


# TODO: Better context management (more functional).
# TODO: graphing
# TODO: ClassDef nodes
# TODO: package structure
# TODO: remove use of self.graph in tests


import ast
import unittest

from pycfa.cf import (
    BREAK,
    CFAnalysis,
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
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(pass_node, {NEXT})
        self.assertEdge(pass_node, NEXT, context[RETURN])

    def test_analyse_single_expr_statement(self):
        code = """\
def f():
    do_something()
"""
        context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Expr)
        self.assertEdges(stmt_node, {NEXT, RAISE})
        self.assertEdge(stmt_node, NEXT, context[RETURN])
        self.assertEdge(stmt_node, RAISE, context[RAISE])

    def test_analyse_assign(self):
        code = """\
def f():
    a = 123
"""
        context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Assign)
        self.assertEdges(stmt_node, {NEXT, RAISE})
        self.assertEdge(stmt_node, NEXT, context[RETURN])
        self.assertEdge(stmt_node, RAISE, context[RAISE])

    def test_analyse_multiple_statements(self):
        code = """\
def f():
    do_something()
    a += do_something_else()
"""
        context, stmt1_node = self._function_context(code)
        self.assertNodetype(stmt1_node, ast.Expr)
        self.assertEdges(stmt1_node, {NEXT, RAISE})
        self.assertEdge(stmt1_node, RAISE, context[RAISE])

        stmt2_node = self.graph.edge(stmt1_node, NEXT)
        self.assertNodetype(stmt2_node, ast.AugAssign)
        self.assertEdges(stmt2_node, {NEXT, RAISE})
        self.assertEdge(stmt2_node, NEXT, context[RETURN])
        self.assertEdge(stmt2_node, RAISE, context[RAISE])

    def test_return_with_no_value(self):
        code = """\
def f():
    return
"""
        context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Return)
        self.assertEdges(stmt_node, {RETURN})
        self.assertEdge(stmt_node, RETURN, context[RETURN])

    def test_return_with_value(self):
        code = """\
def f():
    return None
"""
        context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Return)
        self.assertEdges(stmt_node, {RAISE, RETURN_VALUE})
        self.assertEdge(stmt_node, RAISE, context[RAISE])
        self.assertEdge(stmt_node, RETURN_VALUE, context[RETURN_VALUE])

    def test_raise(self):
        code = """\
def f():
    raise TypeError("don't call me")
"""
        context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Raise)
        self.assertEdges(stmt_node, {RAISE})
        self.assertEdge(stmt_node, RAISE, context[RAISE])

    def test_if(self):
        code = """\
def f():
    if condition:
        a = 123
"""
        context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(if_node, {ELSE, IF, RAISE})
        self.assertEdge(if_node, RAISE, context[RAISE])

        if_branch = self.graph.edge(if_node, IF)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(if_branch, {NEXT, RAISE})
        self.assertEdge(if_branch, RAISE, context[RAISE])
        self.assertEdge(if_branch, NEXT, context[RETURN])
        self.assertEdge(if_node, ELSE, context[RETURN])

    def test_if_else(self):
        code = """\
def f():
    if condition:
        a = 123
    else:
        b = 456
"""
        context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(if_node, {ELSE, IF, RAISE})
        self.assertEdge(if_node, RAISE, context[RAISE])

        if_branch = self.graph.edge(if_node, IF)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(if_branch, {NEXT, RAISE})
        self.assertEdge(if_branch, RAISE, context[RAISE])
        self.assertEdge(if_branch, NEXT, context[RETURN])

        else_branch = self.graph.edge(if_node, ELSE)
        self.assertNodetype(else_branch, ast.Assign)
        self.assertEdges(else_branch, {NEXT, RAISE})
        self.assertEdge(else_branch, RAISE, context[RAISE])
        self.assertEdge(else_branch, NEXT, context[RETURN])

    def test_if_elif_else(self):
        code = """\
def f():
    if some_condition():
        a = something()
    elif some_other_condition():
        a = something_else()
    else:
        a = something_else_again()
"""
        context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(if_node, {ELSE, IF, RAISE})
        self.assertEdge(if_node, RAISE, context[RAISE])

        if_branch = self.graph.edge(if_node, IF)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(if_branch, {NEXT, RAISE})
        self.assertEdge(if_branch, RAISE, context[RAISE])
        self.assertEdge(if_branch, NEXT, context[RETURN])

        elif_node = self.graph.edge(if_node, ELSE)
        self.assertNodetype(elif_node, ast.If)
        self.assertEdges(elif_node, {ELSE, IF, RAISE})
        self.assertEdge(elif_node, RAISE, context[RAISE])

        elif_branch = self.graph.edge(elif_node, IF)
        self.assertNodetype(elif_branch, ast.Assign)
        self.assertEdges(elif_branch, {NEXT, RAISE})
        self.assertEdge(elif_branch, RAISE, context[RAISE])
        self.assertEdge(elif_branch, NEXT, context[RETURN])

        else_branch = self.graph.edge(elif_node, ELSE)
        self.assertNodetype(else_branch, ast.Assign)
        self.assertEdges(else_branch, {NEXT, RAISE})
        self.assertEdge(else_branch, RAISE, context[RAISE])
        self.assertEdge(else_branch, NEXT, context[RETURN])

    def test_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return 123
    else:
        return 456
"""
        context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(if_node, {ELSE, IF, RAISE})
        self.assertEdge(if_node, RAISE, context[RAISE])

        if_branch = self.graph.edge(if_node, IF)
        self.assertNodetype(if_branch, ast.Return)
        self.assertEdges(if_branch, {RAISE, RETURN_VALUE})
        self.assertEdge(if_branch, RETURN_VALUE, context[RETURN_VALUE])
        self.assertEdge(if_branch, RAISE, context[RAISE])

        else_node = self.graph.edge(if_node, ELSE)
        self.assertNodetype(else_node, ast.Return)
        self.assertEdges(else_node, {RAISE, RETURN_VALUE})
        self.assertEdge(else_node, RETURN_VALUE, context[RETURN_VALUE])
        self.assertEdge(else_node, RAISE, context[RAISE])

    def test_plain_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return
    else:
        return
"""
        context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(if_node, {ELSE, IF, RAISE})
        self.assertEdge(if_node, RAISE, context[RAISE])

        if_branch = self.graph.edge(if_node, IF)
        self.assertNodetype(if_branch, ast.Return)
        self.assertEdges(if_branch, {RETURN})
        self.assertEdge(if_branch, RETURN, context[RETURN])

        else_node = self.graph.edge(if_node, ELSE)
        self.assertNodetype(else_node, ast.Return)
        self.assertEdges(else_node, {RETURN})
        self.assertEdge(else_node, RETURN, context[RETURN])

    def test_unreachable_statements(self):
        code = """\
def f():
    do_something()
    return
    do_something_else()
"""
        context, stmt1_node = self._function_context(code)
        self.assertNodetype(stmt1_node, ast.Expr)
        self.assertEdges(stmt1_node, {NEXT, RAISE})
        self.assertEdge(stmt1_node, RAISE, context[RAISE])

        stmt2_node = self.graph.edge(stmt1_node, NEXT)
        self.assertNodetype(stmt2_node, ast.Return)
        self.assertEdges(stmt2_node, {RETURN})
        self.assertEdge(stmt2_node, RETURN, context[RETURN])

    def test_while(self):
        code = """\
def f():
    while some_condition:
        do_something()
"""
        context, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(while_node, RAISE, context[RAISE])
        self.assertEdge(while_node, ELSE, context[RETURN])

        body_node = self.graph.edge(while_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEdge(body_node, RAISE, context[RAISE])
        self.assertEdge(body_node, NEXT, while_node)

    def test_while_else(self):
        code = """\
def f():
    while some_condition:
        do_something()
    else:
        do_no_break_stuff()
"""
        context, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(while_node, RAISE, context[RAISE])

        body_node = self.graph.edge(while_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEdge(body_node, RAISE, context[RAISE])
        self.assertEdge(body_node, NEXT, while_node)

        else_node = self.graph.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEdge(else_node, RAISE, context[RAISE])
        self.assertEdge(else_node, NEXT, context[RETURN])

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
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(while_node, RAISE, context[RAISE])

        test_node = self.graph.edge(while_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(test_node, RAISE, context[RAISE])
        self.assertEdges(test_node, {IF, ELSE, RAISE})

        continue_node = self.graph.edge(test_node, IF)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(continue_node, {CONTINUE})
        self.assertEdge(continue_node, CONTINUE, while_node)

        body_node = self.graph.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEdge(body_node, RAISE, context[RAISE])
        self.assertEdge(body_node, NEXT, while_node)

        else_node = self.graph.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEdge(else_node, RAISE, context[RAISE])
        self.assertEdge(else_node, NEXT, context[RETURN])

    def test_while_with_break(self):
        code = """\
def f():
    while some_condition():
        if not_interesting():
            break
        do_something()
    else:
        do_no_break_stuff()
"""
        context, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(while_node, RAISE, context[RAISE])

        test_node = self.graph.edge(while_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(test_node, RAISE, context[RAISE])
        self.assertEdges(test_node, {IF, ELSE, RAISE})

        break_node = self.graph.edge(test_node, IF)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(break_node, {BREAK})
        self.assertEdge(break_node, BREAK, context[RETURN])

        body_node = self.graph.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEdge(body_node, RAISE, context[RAISE])
        self.assertEdge(body_node, NEXT, while_node)

        else_node = self.graph.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEdge(else_node, RAISE, context[RAISE])
        self.assertEdge(else_node, NEXT, context[RETURN])

    def test_while_with_two_statements(self):
        code = """\
def f():
    while some_condition:
        do_something()
        do_something_else()
"""
        context, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(while_node, RAISE, context[RAISE])
        self.assertEdge(while_node, ELSE, context[RETURN])

        body_node1 = self.graph.edge(while_node, ENTER)
        self.assertNodetype(body_node1, ast.Expr)
        self.assertEdges(body_node1, {NEXT, RAISE})
        self.assertEdge(body_node1, RAISE, context[RAISE])

        body_node2 = self.graph.edge(body_node1, NEXT)
        self.assertNodetype(body_node2, ast.Expr)
        self.assertEdges(body_node2, {NEXT, RAISE})
        self.assertEdge(body_node2, RAISE, context[RAISE])
        self.assertEdge(body_node2, NEXT, while_node)

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
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(for_node, RAISE, context[RAISE])

        test_node = self.graph.edge(for_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(test_node, RAISE, context[RAISE])
        self.assertEdges(test_node, {IF, ELSE, RAISE})

        continue_node = self.graph.edge(test_node, IF)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(continue_node, {CONTINUE})
        self.assertEdge(continue_node, CONTINUE, for_node)

        body_node = self.graph.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEdge(body_node, RAISE, context[RAISE])
        self.assertEdge(body_node, NEXT, for_node)

        else_node = self.graph.edge(for_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEdge(else_node, RAISE, context[RAISE])
        self.assertEdge(else_node, NEXT, context[RETURN])

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
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(for_node, RAISE, context[RAISE])

        test_node = self.graph.edge(for_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(test_node, RAISE, context[RAISE])
        self.assertEdges(test_node, {IF, ELSE, RAISE})

        break_node = self.graph.edge(test_node, IF)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(break_node, {BREAK})
        self.assertEdge(break_node, BREAK, context[RETURN])

        body_node = self.graph.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEdge(body_node, RAISE, context[RAISE])
        self.assertEdge(body_node, NEXT, for_node)

        else_node = self.graph.edge(for_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEdge(else_node, RAISE, context[RAISE])
        self.assertEdge(else_node, NEXT, context[RETURN])

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
        context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(start_node, {ENTER})

        try_node = self.graph.edge(start_node, ENTER)
        self.assertNodetype(try_node, ast.Expr)
        self.assertEdges(try_node, {NEXT, RAISE})

        except1_node = self.graph.edge(try_node, RAISE)
        self.assertNodetype(except1_node, ast.expr)
        self.assertEdges(except1_node, {MATCH, NO_MATCH, RAISE})
        self.assertEdge(except1_node, RAISE, context[RAISE])

        match1_node = self.graph.edge(except1_node, MATCH)
        self.assertNodetype(match1_node, ast.Expr)
        self.assertEdges(match1_node, {NEXT, RAISE})
        self.assertEdge(match1_node, RAISE, context[RAISE])
        self.assertEdge(match1_node, NEXT, context[RETURN])

        match2_node = self.graph.edge(except1_node, NO_MATCH)
        self.assertNodetype(match2_node, ast.Expr)
        self.assertEdges(match2_node, {NEXT, RAISE})
        self.assertEdge(match2_node, RAISE, context[RAISE])
        self.assertEdge(match2_node, NEXT, context[RETURN])

        else_node = self.graph.edge(try_node, NEXT)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(else_node, {NEXT, RAISE})
        self.assertEdge(else_node, RAISE, context[RAISE])
        self.assertEdge(else_node, NEXT, context[RETURN])

    def test_try_except_pass(self):
        code = """\
def f():
    try:
        something_dangerous()
    except:
        pass
"""
        context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(start_node, {ENTER})

        try_node = self.graph.edge(start_node, ENTER)
        self.assertNodetype(try_node, ast.Expr)
        self.assertEdges(try_node, {NEXT, RAISE})
        self.assertEdge(try_node, NEXT, context[RETURN])

        pass_node = self.graph.edge(try_node, RAISE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(pass_node, {NEXT})
        self.assertEdge(pass_node, NEXT, context[RETURN])

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
        context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(start_node, {ENTER})

        try_node = self.graph.edge(start_node, ENTER)
        self.assertNodetype(try_node, ast.Raise)
        self.assertEdges(try_node, {RAISE})

        except_node = self.graph.edge(try_node, RAISE)
        self.assertNodetype(except_node, ast.expr)
        self.assertEdges(except_node, {MATCH, NO_MATCH, RAISE})
        self.assertEdge(except_node, NO_MATCH, context[RAISE])
        self.assertEdge(except_node, RAISE, context[RAISE])

        pass_node = self.graph.edge(except_node, MATCH)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(pass_node, {NEXT})
        self.assertEdge(pass_node, NEXT, context[RETURN])

    def test_try_finally_pass(self):
        code = """\
def f():
    try:
        pass
    finally:
        do_something()
"""
        context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(start_node, {ENTER})

        try_node = self.graph.edge(start_node, ENTER)
        self.assertNodetype(try_node, ast.Pass)
        self.assertEdges(try_node, {NEXT})

        finally_node = self.graph.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEdge(finally_node, RAISE, context[RAISE])
        self.assertEdge(finally_node, NEXT, context[RETURN])

    def test_try_finally_raise(self):
        code = """\
def f():
    try:
        raise ValueError()
    finally:
        do_something()
"""
        context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(start_node, {ENTER})

        try_node = self.graph.edge(start_node, ENTER)
        self.assertNodetype(try_node, ast.Raise)
        self.assertEdges(try_node, {RAISE})

        finally_node = self.graph.edge(try_node, RAISE)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEdge(finally_node, RAISE, context[RAISE])
        self.assertEdge(finally_node, NEXT, context[RAISE])

    def test_try_finally_return(self):
        code = """\
def f():
    try:
        return
    finally:
        do_something()
"""
        context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(start_node, {ENTER})

        try_node = self.graph.edge(start_node, ENTER)
        self.assertNodetype(try_node, ast.Return)
        self.assertEdges(try_node, {RETURN})

        finally_node = self.graph.edge(try_node, RETURN)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEdge(finally_node, RAISE, context[RAISE])
        self.assertEdge(finally_node, NEXT, context[RETURN])

    def test_try_finally_return_value(self):
        code = """\
def f():
    try:
        return "abc"
    finally:
        do_something()
"""
        context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(start_node, {ENTER})

        try_node = self.graph.edge(start_node, ENTER)
        self.assertNodetype(try_node, ast.Return)
        self.assertEdges(try_node, {RAISE, RETURN_VALUE})

        finally_node = self.graph.edge(try_node, RETURN_VALUE)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEdge(finally_node, RAISE, context[RAISE])
        self.assertEdge(finally_node, NEXT, context[RETURN_VALUE])

        finally2_node = self.graph.edge(try_node, RAISE)
        self.assertNodetype(finally2_node, ast.Expr)
        self.assertEdges(finally2_node, {NEXT, RAISE})
        self.assertEdge(finally2_node, RAISE, context[RAISE])
        self.assertEdge(finally2_node, NEXT, context[RAISE])

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
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(for_node, ELSE, context[RETURN])
        self.assertEdge(for_node, RAISE, context[RAISE])

        try_node = self.graph.edge(for_node, ENTER)
        self.assertNodetype(try_node, ast.Try)
        self.assertEdges(try_node, {ENTER})

        break_node = self.graph.edge(try_node, ENTER)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(break_node, {BREAK})

        finally_node = self.graph.edge(break_node, BREAK)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEdge(finally_node, RAISE, context[RAISE])
        self.assertEdge(finally_node, NEXT, context[RETURN])

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
        self.assertEdge(for_node, ELSE, context[RETURN])
        self.assertEdge(for_node, RAISE, context[RAISE])

        try_node = self.graph.edge(for_node, ENTER)
        self.assertEdges(try_node, {ENTER})

        continue_node = self.graph.edge(try_node, ENTER)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(continue_node, {CONTINUE})

        finally_node = self.graph.edge(continue_node, CONTINUE)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEdge(finally_node, RAISE, context[RAISE])
        self.assertEdge(finally_node, NEXT, for_node)

    def test_return_value_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return some_value()
"""
        context, try_node = self._function_context(code)
        self.assertEdges(try_node, {ENTER})

        raise_node = self.graph.edge(try_node, ENTER)
        self.assertEdges(raise_node, {RAISE})

        return_node = self.graph.edge(raise_node, RAISE)
        self.assertEdges(return_node, {RAISE, RETURN_VALUE})
        self.assertEdge(return_node, RETURN_VALUE, context[RETURN_VALUE])
        self.assertEdge(return_node, RAISE, context[RAISE])

    def test_return_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return
"""
        context, try_node = self._function_context(code)
        self.assertEdges(try_node, {ENTER})

        raise_node = self.graph.edge(try_node, ENTER)
        self.assertEdges(raise_node, {RAISE})

        return_node = self.graph.edge(raise_node, RAISE)
        self.assertEdges(return_node, {RETURN})
        self.assertEdge(return_node, RETURN, context[RETURN])

    def test_raise_in_finally(self):
        code = """\
def f():
    try:
        pass
    finally:
        raise SomeException()
"""
        context, try_node = self._function_context(code)
        self.assertEdges(try_node, {ENTER})

        pass_node = self.graph.edge(try_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(pass_node, {NEXT})

        raise_node = self.graph.edge(pass_node, NEXT)
        self.assertNodetype(raise_node, ast.Raise)
        self.assertEdges(raise_node, {RAISE})
        self.assertEdge(raise_node, RAISE, context[RAISE])

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
        self.assertEdge(for_node, ELSE, context[RETURN])
        self.assertEdge(for_node, RAISE, context[RAISE])

        try_node = self.graph.edge(for_node, ENTER)
        self.assertEdges(try_node, {ENTER})

        return_node = self.graph.edge(try_node, ENTER)
        self.assertEdges(return_node, {RETURN})

        break_node = self.graph.edge(return_node, RETURN)
        self.assertEdges(break_node, {BREAK})
        self.assertEdge(break_node, BREAK, context[RETURN])

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
        self.assertEdge(for_node, ELSE, context[RETURN])
        self.assertEdge(for_node, RAISE, context[RAISE])

        try_node = self.graph.edge(for_node, ENTER)
        self.assertEdges(try_node, {ENTER})

        raise_node = self.graph.edge(try_node, ENTER)
        self.assertEdges(raise_node, {RAISE})

        continue_node = self.graph.edge(raise_node, RAISE)
        self.assertEdges(continue_node, {CONTINUE})
        self.assertEdge(continue_node, CONTINUE, for_node)

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
        self.assertEdge(for_node, ELSE, context[RETURN])
        self.assertEdge(for_node, RAISE, context[RAISE])

        try_node = self.graph.edge(for_node, ENTER)
        self.assertEdges(try_node, {ENTER})

        raise_node = self.graph.edge(try_node, ENTER)
        self.assertEdges(raise_node, {RAISE})

        continue_node = self.graph.edge(raise_node, RAISE)
        self.assertEdges(continue_node, {CONTINUE})
        self.assertEdge(continue_node, CONTINUE, for_node)

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
        self.assertEdge(for_node, ELSE, context[RETURN])
        self.assertEdge(for_node, RAISE, context[RAISE])

        try_node = self.graph.edge(for_node, ENTER)
        self.assertEdges(try_node, {ENTER})

        raise_node = self.graph.edge(try_node, ENTER)
        self.assertEdges(raise_node, {RAISE})

        continue_node = self.graph.edge(raise_node, RAISE)
        self.assertEdges(continue_node, {CONTINUE})

        finally_node = self.graph.edge(continue_node, CONTINUE)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEdge(finally_node, RAISE, context[RAISE])
        self.assertEdge(finally_node, NEXT, for_node)

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
        self.assertEdge(for_node, ELSE, context[RETURN])
        self.assertEdge(for_node, RAISE, context[RAISE])

        try_node = self.graph.edge(for_node, ENTER)
        self.assertEdges(try_node, {ENTER})

        raise_node = self.graph.edge(try_node, ENTER)
        self.assertEdges(raise_node, {RAISE})

        except_node = self.graph.edge(raise_node, RAISE)
        self.assertEdges(except_node, {MATCH, NO_MATCH, RAISE})
        self.assertEdge(
            except_node, RAISE, self.graph.edge(except_node, NO_MATCH)
        )

        finally_raise_node = self.graph.edge(except_node, RAISE)
        self.assertEdges(finally_raise_node, {NEXT, RAISE})
        self.assertEdge(finally_raise_node, RAISE, context[RAISE])
        self.assertEdge(finally_raise_node, NEXT, context[RAISE])

        break_node = self.graph.edge(except_node, MATCH)
        self.assertEdges(break_node, {BREAK})

        finally_node = self.graph.edge(break_node, BREAK)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEdge(finally_node, RAISE, context[RAISE])
        self.assertEdge(finally_node, NEXT, context[RETURN])

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
        self.assertEdge(for_node, ELSE, context[RETURN])
        self.assertEdge(for_node, RAISE, context[RAISE])

        try_node = self.graph.edge(for_node, ENTER)
        self.assertEdges(try_node, {ENTER})

        do_node = self.graph.edge(try_node, ENTER)
        self.assertEdges(do_node, {NEXT, RAISE})

        pass_node = self.graph.edge(do_node, RAISE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(pass_node, {NEXT})

        finally1_node = self.graph.edge(pass_node, NEXT)
        self.assertEdges(finally1_node, {NEXT, RAISE})
        self.assertEdge(finally1_node, RAISE, context[RAISE])
        self.assertEdge(finally1_node, NEXT, for_node)

        else_node = self.graph.edge(do_node, NEXT)
        self.assertEdges(else_node, {RETURN})

        finally_node = self.graph.edge(else_node, RETURN)
        self.assertEdges(finally_node, {NEXT, RAISE})
        self.assertEdge(finally_node, RAISE, context[RAISE])
        self.assertEdge(finally_node, NEXT, context[RETURN])

    def test_finally_paths_combined(self):
        # We potentially generate multiple paths for a finally block,
        # but paths with the same NEXT should be combined.
        code = """\
def f():
    try:
        do_something()
    except:
        handle_error()
    else:
        return
    finally:
        do_cleanup()
"""
        context, try_node = self._function_context(code)

        # The 'return' in the else branch should lead to the same place
        # as the handle_exception() success in the except branch.
        do_node = self.graph.edge(try_node, ENTER)
        raised_node = self.graph.edge(do_node, RAISE)
        ok_node = self.graph.edge(do_node, NEXT)

        raised_next = self.graph.edge(raised_node, NEXT)
        ok_next = self.graph.edge(ok_node, RETURN)

        self.assertEqual(raised_next, ok_next)

    def test_empty_module(self):
        code = ""
        context, enter_node = self._module_context(code)
        self.assertEqual(enter_node, context[NEXT])

    def test_just_pass(self):
        code = """\
pass
"""
        context, enter_node = self._module_context(code)
        self.assertNodetype(enter_node, ast.Pass)
        self.assertEdges(enter_node, {NEXT})
        self.assertEdge(enter_node, NEXT, context[NEXT])

    def test_statements_outside_function(self):
        code = """\
a = calculate()
try:
    something()
except:
    pass
"""
        context, assign_node = self._module_context(code)
        self.assertNodetype(assign_node, ast.Assign)
        self.assertEdges(assign_node, {NEXT, RAISE})
        self.assertEdge(assign_node, RAISE, context[RAISE])

        try_node = self.graph.edge(assign_node, NEXT)
        self.assertNodetype(try_node, ast.Try)
        self.assertEdges(try_node, {ENTER})

        do_node = self.graph.edge(try_node, ENTER)
        self.assertNodetype(do_node, ast.Expr)
        self.assertEdges(do_node, {NEXT, RAISE})
        self.assertEdge(do_node, NEXT, context[NEXT])

        pass_node = self.graph.edge(do_node, RAISE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(pass_node, {NEXT})
        self.assertEdge(pass_node, NEXT, context[NEXT])

    def test_with(self):
        code = """\
with some_cm() as name:
    do_something()
"""
        context, with_node = self._module_context(code)
        self.assertNodetype(with_node, ast.With)
        self.assertEdges(with_node, {ENTER, RAISE})
        self.assertEdge(with_node, RAISE, context[RAISE])

        body_node = self.graph.edge(with_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(body_node, {NEXT, RAISE})
        self.assertEdge(body_node, RAISE, context[RAISE])
        self.assertEdge(body_node, NEXT, context[NEXT])

    def test_async_for(self):
        code = """\
async def f():
    async for x in g():
        yield x*x
"""
        context, for_node = self._function_context(code)
        self.assertNodetype(for_node, ast.AsyncFor)
        self.assertEdges(for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(for_node, ELSE, context[RETURN])
        self.assertEdge(for_node, RAISE, context[RAISE])

        yield_node = self.graph.edge(for_node, ENTER)
        self.assertNodetype(yield_node, ast.Expr)
        self.assertEdges(yield_node, {NEXT, RAISE})
        self.assertEdge(yield_node, NEXT, for_node)
        self.assertEdge(yield_node, RAISE, context[RAISE])

    def test_async_with(self):
        code = """\
async def f():
    async with my_async_context():
        pass
"""
        context, with_node = self._function_context(code)
        self.assertNodetype(with_node, ast.AsyncWith)
        self.assertEdges(with_node, {ENTER, RAISE})
        self.assertEdge(with_node, RAISE, context[RAISE])

        pass_node = self.graph.edge(with_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(pass_node, {NEXT})
        self.assertEdge(pass_node, NEXT, context[RETURN])

    def test_global(self):
        code = """\
def f():
    global bob
"""
        context, node = self._function_context(code)
        self.assertEqual(node, context[RETURN])

    def test_nonlocal(self):
        code = """\
def f(bob):
    def g():
        nonlocal bob
"""
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        (function_node,) = module_node.body
        (inner_function,) = function_node.body

        graph = CFAnalysis.from_function(inner_function)
        context = graph.context
        node = context[ENTER]
        self.assertEqual(node, context[RETURN])

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
assert 2 is not 3
x : int = 2
async def beckett():
    await godot()
"""
        context, node = self._module_context(code)

        for _ in range(9):
            self.assertNodetype(node, ast.stmt)
            self.assertEdges(node, {NEXT, RAISE})
            self.assertEdge(node, RAISE, context[RAISE])
            node = self.graph.edge(node, NEXT)

        self.assertEqual(node, context[NEXT])

    # Assertions

    def assertEdges(self, node, edges):
        """
        Assert that the outward edges from a node have the given names.
        """
        self.assertEqual(self.graph.edge_labels(node), edges)

    def assertEdge(self, source, label, target):
        """
        Assert that the given edge from the given source maps to the
        given target.
        """
        self.assertEqual(self.graph.edge(source, label), target)

    def assertNodetype(self, node, nodetype):
        """
        Assert that the given control-flow graph node is associated
        to an ast node of the given type.
        """
        self.assertIsInstance(node.ast_node, nodetype)

    # Helper methods

    def _function_context(self, code):
        (function_node,) = compile(
            code, "test_cf", "exec", ast.PyCF_ONLY_AST
        ).body
        self.assertIsInstance(
            function_node, (ast.AsyncFunctionDef, ast.FunctionDef)
        )

        graph = CFAnalysis.from_function(function_node)
        context = graph.context
        self.graph = graph
        self.assertEqual(
            sorted(context.keys()), [ENTER, RAISE, RETURN, RETURN_VALUE],
        )
        self.assertEdges(context[RAISE], set())
        self.assertEdges(context[RETURN], set())
        self.assertEdges(context[RETURN_VALUE], set())

        return context, context[ENTER]

    def _module_context(self, code):
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        self.assertIsInstance(module_node, ast.Module)

        graph = CFAnalysis.from_module(module_node)
        context = graph.context
        self.graph = graph
        self.assertEqual(
            sorted(context.keys()), [ENTER, NEXT, RAISE],
        )
        self.assertEdges(context[RAISE], set())
        self.assertEdges(context[NEXT], set())

        return context, context[ENTER]
