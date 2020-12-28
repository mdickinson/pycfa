# Copyright 2020 Mark Dickinson. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Analyse control flow for a piece of Python code.

Aid in detection of things like unreachable code.
"""

# TODO: add node types, and document edge labels for each node type?
# TODO: Better context management (more functional).


import ast
import unittest

from pycfa.cf import (
    CFAnalysis,
    ELSE,
    ENTER,
    ENTERC,
    NEXT,
    NEXTC,
    RAISE,
    RAISEC,
    RETURN,
    RETURN_VALUE,
)


class TestCFAnalysis(unittest.TestCase):
    def test_analyse_noop_function(self):
        code = """\
def f():
    pass
"""
        graph, context, pass_node = self._function_context(code)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(graph, pass_node, {NEXT})
        self.assertEdge(graph, pass_node, NEXT, context[RETURN])

    def test_analyse_single_expr_statement(self):
        code = """\
def f():
    do_something()
"""
        graph, context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Expr)
        self.assertEdges(graph, stmt_node, {NEXT, RAISE})
        self.assertEdge(graph, stmt_node, NEXT, context[RETURN])
        self.assertEdge(graph, stmt_node, RAISE, context[RAISEC])

    def test_analyse_assign(self):
        code = """\
def f():
    a = 123
"""
        graph, context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Assign)
        self.assertEdges(graph, stmt_node, {NEXT, RAISE})
        self.assertEdge(graph, stmt_node, NEXT, context[RETURN])
        self.assertEdge(graph, stmt_node, RAISE, context[RAISEC])

    def test_analyse_multiple_statements(self):
        code = """\
def f():
    do_something()
    a += do_something_else()
"""
        graph, context, stmt1_node = self._function_context(code)
        self.assertNodetype(stmt1_node, ast.Expr)
        self.assertEdges(graph, stmt1_node, {NEXT, RAISE})
        self.assertEdge(graph, stmt1_node, RAISE, context[RAISEC])

        stmt2_node = graph.edge(stmt1_node, NEXT)
        self.assertNodetype(stmt2_node, ast.AugAssign)
        self.assertEdges(graph, stmt2_node, {NEXT, RAISE})
        self.assertEdge(graph, stmt2_node, NEXT, context[RETURN])
        self.assertEdge(graph, stmt2_node, RAISE, context[RAISEC])

    def test_return_with_no_value(self):
        code = """\
def f():
    return
"""
        graph, context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Return)
        self.assertEdges(graph, stmt_node, {NEXT})
        self.assertEdge(graph, stmt_node, NEXT, context[RETURN])

    def test_return_with_value(self):
        code = """\
def f():
    return None
"""
        graph, context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Return)
        self.assertEdges(graph, stmt_node, {NEXT, RAISE})
        self.assertEdge(graph, stmt_node, RAISE, context[RAISEC])
        self.assertEdge(graph, stmt_node, NEXT, context[RETURN_VALUE])

    def test_raise(self):
        code = """\
def f():
    raise TypeError("don't call me")
"""
        graph, context, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Raise)
        self.assertEdges(graph, stmt_node, {RAISE})
        self.assertEdge(graph, stmt_node, RAISE, context[RAISEC])

    def test_if(self):
        code = """\
def f():
    if condition:
        a = 123
"""
        graph, context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(graph, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, if_node, RAISE, context[RAISEC])

        if_branch = graph.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(graph, if_branch, {NEXT, RAISE})
        self.assertEdge(graph, if_branch, RAISE, context[RAISEC])
        self.assertEdge(graph, if_branch, NEXT, context[RETURN])
        self.assertEdge(graph, if_node, ELSE, context[RETURN])

    def test_if_else(self):
        code = """\
def f():
    if condition:
        a = 123
    else:
        b = 456
"""
        graph, context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(graph, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, if_node, RAISE, context[RAISEC])

        if_branch = graph.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(graph, if_branch, {NEXT, RAISE})
        self.assertEdge(graph, if_branch, RAISE, context[RAISEC])
        self.assertEdge(graph, if_branch, NEXT, context[RETURN])

        else_branch = graph.edge(if_node, ELSE)
        self.assertNodetype(else_branch, ast.Assign)
        self.assertEdges(graph, else_branch, {NEXT, RAISE})
        self.assertEdge(graph, else_branch, RAISE, context[RAISEC])
        self.assertEdge(graph, else_branch, NEXT, context[RETURN])

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
        graph, context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(graph, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, if_node, RAISE, context[RAISEC])

        if_branch = graph.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(graph, if_branch, {NEXT, RAISE})
        self.assertEdge(graph, if_branch, RAISE, context[RAISEC])
        self.assertEdge(graph, if_branch, NEXT, context[RETURN])

        elif_node = graph.edge(if_node, ELSE)
        self.assertNodetype(elif_node, ast.If)
        self.assertEdges(graph, elif_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, elif_node, RAISE, context[RAISEC])

        elif_branch = graph.edge(elif_node, ENTER)
        self.assertNodetype(elif_branch, ast.Assign)
        self.assertEdges(graph, elif_branch, {NEXT, RAISE})
        self.assertEdge(graph, elif_branch, RAISE, context[RAISEC])
        self.assertEdge(graph, elif_branch, NEXT, context[RETURN])

        else_branch = graph.edge(elif_node, ELSE)
        self.assertNodetype(else_branch, ast.Assign)
        self.assertEdges(graph, else_branch, {NEXT, RAISE})
        self.assertEdge(graph, else_branch, RAISE, context[RAISEC])
        self.assertEdge(graph, else_branch, NEXT, context[RETURN])

    def test_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return 123
    else:
        return 456
"""
        graph, context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(graph, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, if_node, RAISE, context[RAISEC])

        if_branch = graph.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Return)
        self.assertEdges(graph, if_branch, {NEXT, RAISE})
        self.assertEdge(graph, if_branch, NEXT, context[RETURN_VALUE])
        self.assertEdge(graph, if_branch, RAISE, context[RAISEC])

        else_node = graph.edge(if_node, ELSE)
        self.assertNodetype(else_node, ast.Return)
        self.assertEdges(graph, else_node, {NEXT, RAISE})
        self.assertEdge(graph, else_node, NEXT, context[RETURN_VALUE])
        self.assertEdge(graph, else_node, RAISE, context[RAISEC])

    def test_plain_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return
    else:
        return
"""
        graph, context, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(graph, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, if_node, RAISE, context[RAISEC])

        if_branch = graph.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Return)
        self.assertEdges(graph, if_branch, {NEXT})
        self.assertEdge(graph, if_branch, NEXT, context[RETURN])

        else_node = graph.edge(if_node, ELSE)
        self.assertNodetype(else_node, ast.Return)
        self.assertEdges(graph, else_node, {NEXT})
        self.assertEdge(graph, else_node, NEXT, context[RETURN])

    def test_unreachable_statements(self):
        code = """\
def f():
    do_something()
    return
    do_something_else()
"""
        graph, context, stmt1_node = self._function_context(code)
        self.assertNodetype(stmt1_node, ast.Expr)
        self.assertEdges(graph, stmt1_node, {NEXT, RAISE})
        self.assertEdge(graph, stmt1_node, RAISE, context[RAISEC])

        stmt2_node = graph.edge(stmt1_node, NEXT)
        self.assertNodetype(stmt2_node, ast.Return)
        self.assertEdges(graph, stmt2_node, {NEXT})
        self.assertEdge(graph, stmt2_node, NEXT, context[RETURN])

    def test_while(self):
        code = """\
def f():
    while some_condition:
        do_something()
"""
        graph, context, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(graph, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, while_node, RAISE, context[RAISEC])
        self.assertEdge(graph, while_node, ELSE, context[RETURN])

        body_node = graph.edge(while_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(graph, body_node, {NEXT, RAISE})
        self.assertEdge(graph, body_node, RAISE, context[RAISEC])
        self.assertEdge(graph, body_node, NEXT, while_node)

    def test_while_else(self):
        code = """\
def f():
    while some_condition:
        do_something()
    else:
        do_no_break_stuff()
"""
        graph, context, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(graph, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, while_node, RAISE, context[RAISEC])

        body_node = graph.edge(while_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(graph, body_node, {NEXT, RAISE})
        self.assertEdge(graph, body_node, RAISE, context[RAISEC])
        self.assertEdge(graph, body_node, NEXT, while_node)

        else_node = graph.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(graph, else_node, {NEXT, RAISE})
        self.assertEdge(graph, else_node, RAISE, context[RAISEC])
        self.assertEdge(graph, else_node, NEXT, context[RETURN])

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
        graph, context, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(graph, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, while_node, RAISE, context[RAISEC])

        test_node = graph.edge(while_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(graph, test_node, RAISE, context[RAISEC])
        self.assertEdges(graph, test_node, {ENTER, ELSE, RAISE})

        continue_node = graph.edge(test_node, ENTER)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(graph, continue_node, {NEXT})
        self.assertEdge(graph, continue_node, NEXT, while_node)

        body_node = graph.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(graph, body_node, {NEXT, RAISE})
        self.assertEdge(graph, body_node, RAISE, context[RAISEC])
        self.assertEdge(graph, body_node, NEXT, while_node)

        else_node = graph.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(graph, else_node, {NEXT, RAISE})
        self.assertEdge(graph, else_node, RAISE, context[RAISEC])
        self.assertEdge(graph, else_node, NEXT, context[RETURN])

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
        graph, context, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(graph, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, while_node, RAISE, context[RAISEC])

        test_node = graph.edge(while_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(graph, test_node, RAISE, context[RAISEC])
        self.assertEdges(graph, test_node, {ENTER, ELSE, RAISE})

        break_node = graph.edge(test_node, ENTER)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(graph, break_node, {NEXT})
        self.assertEdge(graph, break_node, NEXT, context[RETURN])

        body_node = graph.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(graph, body_node, {NEXT, RAISE})
        self.assertEdge(graph, body_node, RAISE, context[RAISEC])
        self.assertEdge(graph, body_node, NEXT, while_node)

        else_node = graph.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(graph, else_node, {NEXT, RAISE})
        self.assertEdge(graph, else_node, RAISE, context[RAISEC])
        self.assertEdge(graph, else_node, NEXT, context[RETURN])

    def test_while_with_two_statements(self):
        code = """\
def f():
    while some_condition:
        do_something()
        do_something_else()
"""
        graph, context, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(graph, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, while_node, RAISE, context[RAISEC])
        self.assertEdge(graph, while_node, ELSE, context[RETURN])

        body_node1 = graph.edge(while_node, ENTER)
        self.assertNodetype(body_node1, ast.Expr)
        self.assertEdges(graph, body_node1, {NEXT, RAISE})
        self.assertEdge(graph, body_node1, RAISE, context[RAISEC])

        body_node2 = graph.edge(body_node1, NEXT)
        self.assertNodetype(body_node2, ast.Expr)
        self.assertEdges(graph, body_node2, {NEXT, RAISE})
        self.assertEdge(graph, body_node2, RAISE, context[RAISEC])
        self.assertEdge(graph, body_node2, NEXT, while_node)

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
        graph, context, for_node = self._function_context(code)
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        test_node = graph.edge(for_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(graph, test_node, RAISE, context[RAISEC])
        self.assertEdges(graph, test_node, {ENTER, ELSE, RAISE})

        continue_node = graph.edge(test_node, ENTER)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(graph, continue_node, {NEXT})
        self.assertEdge(graph, continue_node, NEXT, for_node)

        body_node = graph.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(graph, body_node, {NEXT, RAISE})
        self.assertEdge(graph, body_node, RAISE, context[RAISEC])
        self.assertEdge(graph, body_node, NEXT, for_node)

        else_node = graph.edge(for_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(graph, else_node, {NEXT, RAISE})
        self.assertEdge(graph, else_node, RAISE, context[RAISEC])
        self.assertEdge(graph, else_node, NEXT, context[RETURN])

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
        graph, context, for_node = self._function_context(code)
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        test_node = graph.edge(for_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(graph, test_node, RAISE, context[RAISEC])
        self.assertEdges(graph, test_node, {ENTER, ELSE, RAISE})

        break_node = graph.edge(test_node, ENTER)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(graph, break_node, {NEXT})
        self.assertEdge(graph, break_node, NEXT, context[RETURN])

        body_node = graph.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(graph, body_node, {NEXT, RAISE})
        self.assertEdge(graph, body_node, RAISE, context[RAISEC])
        self.assertEdge(graph, body_node, NEXT, for_node)

        else_node = graph.edge(for_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(graph, else_node, {NEXT, RAISE})
        self.assertEdge(graph, else_node, RAISE, context[RAISEC])
        self.assertEdge(graph, else_node, NEXT, context[RETURN])

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
        graph, context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(graph, start_node, {NEXT})

        try_node = graph.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Expr)
        self.assertEdges(graph, try_node, {NEXT, RAISE})

        except1_node = graph.edge(try_node, RAISE)
        self.assertNodetype(except1_node, ast.expr)
        self.assertEdges(graph, except1_node, {ENTER, ELSE, RAISE})
        self.assertEdge(graph, except1_node, RAISE, context[RAISEC])

        match1_node = graph.edge(except1_node, ENTER)
        self.assertNodetype(match1_node, ast.Expr)
        self.assertEdges(graph, match1_node, {NEXT, RAISE})
        self.assertEdge(graph, match1_node, RAISE, context[RAISEC])
        self.assertEdge(graph, match1_node, NEXT, context[RETURN])

        match2_node = graph.edge(except1_node, ELSE)
        self.assertNodetype(match2_node, ast.Expr)
        self.assertEdges(graph, match2_node, {NEXT, RAISE})
        self.assertEdge(graph, match2_node, RAISE, context[RAISEC])
        self.assertEdge(graph, match2_node, NEXT, context[RETURN])

        else_node = graph.edge(try_node, NEXT)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(graph, else_node, {NEXT, RAISE})
        self.assertEdge(graph, else_node, RAISE, context[RAISEC])
        self.assertEdge(graph, else_node, NEXT, context[RETURN])

    def test_try_except_pass(self):
        code = """\
def f():
    try:
        something_dangerous()
    except:
        pass
"""
        graph, context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(graph, start_node, {NEXT})

        try_node = graph.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Expr)
        self.assertEdges(graph, try_node, {NEXT, RAISE})
        self.assertEdge(graph, try_node, NEXT, context[RETURN])

        pass_node = graph.edge(try_node, RAISE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(graph, pass_node, {NEXT})
        self.assertEdge(graph, pass_node, NEXT, context[RETURN])

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
        graph, context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(graph, start_node, {NEXT})

        try_node = graph.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Raise)
        self.assertEdges(graph, try_node, {RAISE})

        except_node = graph.edge(try_node, RAISE)
        self.assertNodetype(except_node, ast.expr)
        self.assertEdges(graph, except_node, {ENTER, ELSE, RAISE})
        self.assertEdge(graph, except_node, ELSE, context[RAISEC])
        self.assertEdge(graph, except_node, RAISE, context[RAISEC])

        pass_node = graph.edge(except_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(graph, pass_node, {NEXT})
        self.assertEdge(graph, pass_node, NEXT, context[RETURN])

    def test_try_finally_pass(self):
        code = """\
def f():
    try:
        pass
    finally:
        do_something()
"""
        graph, context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(graph, start_node, {NEXT})

        try_node = graph.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Pass)
        self.assertEdges(graph, try_node, {NEXT})

        finally_node = graph.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(graph, finally_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_node, NEXT, context[RETURN])

    def test_try_finally_raise(self):
        code = """\
def f():
    try:
        raise ValueError()
    finally:
        do_something()
"""
        graph, context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(graph, start_node, {NEXT})

        try_node = graph.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Raise)
        self.assertEdges(graph, try_node, {RAISE})

        finally_node = graph.edge(try_node, RAISE)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(graph, finally_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_node, NEXT, context[RAISEC])

    def test_try_finally_return(self):
        code = """\
def f():
    try:
        return
    finally:
        do_something()
"""
        graph, context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(graph, start_node, {NEXT})

        try_node = graph.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Return)
        self.assertEdges(graph, try_node, {NEXT})

        finally_node = graph.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(graph, finally_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_node, NEXT, context[RETURN])

    def test_try_finally_return_value(self):
        code = """\
def f():
    try:
        return "abc"
    finally:
        do_something()
"""
        graph, context, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(graph, start_node, {NEXT})

        try_node = graph.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Return)
        self.assertEdges(graph, try_node, {NEXT, RAISE})

        finally_node = graph.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(graph, finally_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_node, NEXT, context[RETURN_VALUE])

        finally2_node = graph.edge(try_node, RAISE)
        self.assertNodetype(finally2_node, ast.Expr)
        self.assertEdges(graph, finally2_node, {NEXT, RAISE})
        self.assertEdge(graph, finally2_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally2_node, NEXT, context[RAISEC])

    def test_try_finally_break(self):
        code = """\
def f():
    for item in items:
        try:
            break
        finally:
            do_something()
"""
        graph, context, for_node = self._function_context(code)
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, ELSE, context[RETURN])
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        try_node = graph.edge(for_node, ENTER)
        self.assertNodetype(try_node, ast.Try)
        self.assertEdges(graph, try_node, {NEXT})

        break_node = graph.edge(try_node, NEXT)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(graph, break_node, {NEXT})

        finally_node = graph.edge(break_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(graph, finally_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_node, NEXT, context[RETURN])

    def test_try_finally_continue(self):
        code = """\
def f():
    for item in items:
        try:
            continue
        finally:
            do_something()
"""
        graph, context, for_node = self._function_context(code)

        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, ELSE, context[RETURN])
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        try_node = graph.edge(for_node, ENTER)
        self.assertEdges(graph, try_node, {NEXT})

        continue_node = graph.edge(try_node, NEXT)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(graph, continue_node, {NEXT})

        finally_node = graph.edge(continue_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(graph, finally_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_node, NEXT, for_node)

    def test_return_value_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return some_value()
"""
        graph, context, try_node = self._function_context(code)
        self.assertEdges(graph, try_node, {NEXT})

        raise_node = graph.edge(try_node, NEXT)
        self.assertEdges(graph, raise_node, {RAISE})

        return_node = graph.edge(raise_node, RAISE)
        self.assertEdges(graph, return_node, {NEXT, RAISE})
        self.assertEdge(graph, return_node, NEXT, context[RETURN_VALUE])
        self.assertEdge(graph, return_node, RAISE, context[RAISEC])

    def test_return_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return
"""
        graph, context, try_node = self._function_context(code)
        self.assertEdges(graph, try_node, {NEXT})

        raise_node = graph.edge(try_node, NEXT)
        self.assertEdges(graph, raise_node, {RAISE})

        return_node = graph.edge(raise_node, RAISE)
        self.assertEdges(graph, return_node, {NEXT})
        self.assertEdge(graph, return_node, NEXT, context[RETURN])

    def test_raise_in_finally(self):
        code = """\
def f():
    try:
        pass
    finally:
        raise SomeException()
"""
        graph, context, try_node = self._function_context(code)
        self.assertEdges(graph, try_node, {NEXT})

        pass_node = graph.edge(try_node, NEXT)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(graph, pass_node, {NEXT})

        raise_node = graph.edge(pass_node, NEXT)
        self.assertNodetype(raise_node, ast.Raise)
        self.assertEdges(graph, raise_node, {RAISE})
        self.assertEdge(graph, raise_node, RAISE, context[RAISEC])

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
        graph, context, for_node = self._function_context(code)

        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, ELSE, context[RETURN])
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        try_node = graph.edge(for_node, ENTER)
        self.assertEdges(graph, try_node, {NEXT})

        return_node = graph.edge(try_node, NEXT)
        self.assertEdges(graph, return_node, {NEXT})

        break_node = graph.edge(return_node, NEXT)
        self.assertEdges(graph, break_node, {NEXT})
        self.assertEdge(graph, break_node, NEXT, context[RETURN])

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
        graph, context, for_node = self._function_context(code)

        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, ELSE, context[RETURN])
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        try_node = graph.edge(for_node, ENTER)
        self.assertEdges(graph, try_node, {NEXT})

        raise_node = graph.edge(try_node, NEXT)
        self.assertEdges(graph, raise_node, {RAISE})

        continue_node = graph.edge(raise_node, RAISE)
        self.assertEdges(graph, continue_node, {NEXT})
        self.assertEdge(graph, continue_node, NEXT, for_node)

    def test_continue_in_except_no_finally(self):
        code = """\
def f():
    for item in item_factory():
        try:
            raise SomeException()
        except:
            continue
"""
        graph, context, for_node = self._function_context(code)

        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, ELSE, context[RETURN])
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        try_node = graph.edge(for_node, ENTER)
        self.assertEdges(graph, try_node, {NEXT})

        raise_node = graph.edge(try_node, NEXT)
        self.assertEdges(graph, raise_node, {RAISE})

        continue_node = graph.edge(raise_node, RAISE)
        self.assertEdges(graph, continue_node, {NEXT})
        self.assertEdge(graph, continue_node, NEXT, for_node)

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
        graph, context, for_node = self._function_context(code)

        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, ELSE, context[RETURN])
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        try_node = graph.edge(for_node, ENTER)
        self.assertEdges(graph, try_node, {NEXT})

        raise_node = graph.edge(try_node, NEXT)
        self.assertEdges(graph, raise_node, {RAISE})

        continue_node = graph.edge(raise_node, RAISE)
        self.assertEdges(graph, continue_node, {NEXT})

        finally_node = graph.edge(continue_node, NEXT)
        self.assertEdges(graph, finally_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_node, NEXT, for_node)

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
        graph, context, for_node = self._function_context(code)

        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, ELSE, context[RETURN])
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        try_node = graph.edge(for_node, ENTER)
        self.assertEdges(graph, try_node, {NEXT})

        raise_node = graph.edge(try_node, NEXT)
        self.assertEdges(graph, raise_node, {RAISE})

        except_node = graph.edge(raise_node, RAISE)
        self.assertEdges(graph, except_node, {ENTER, ELSE, RAISE})
        self.assertEdge(
            graph, except_node, RAISE, graph.edge(except_node, ELSE)
        )

        finally_raise_node = graph.edge(except_node, RAISE)
        self.assertEdges(graph, finally_raise_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_raise_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_raise_node, NEXT, context[RAISEC])

        break_node = graph.edge(except_node, ENTER)
        self.assertEdges(graph, break_node, {NEXT})

        finally_node = graph.edge(break_node, NEXT)
        self.assertEdges(graph, finally_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_node, NEXT, context[RETURN])

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
        graph, context, for_node = self._function_context(code)

        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, ELSE, context[RETURN])
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        try_node = graph.edge(for_node, ENTER)
        self.assertEdges(graph, try_node, {NEXT})

        do_node = graph.edge(try_node, NEXT)
        self.assertEdges(graph, do_node, {NEXT, RAISE})

        pass_node = graph.edge(do_node, RAISE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(graph, pass_node, {NEXT})

        finally1_node = graph.edge(pass_node, NEXT)
        self.assertEdges(graph, finally1_node, {NEXT, RAISE})
        self.assertEdge(graph, finally1_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally1_node, NEXT, for_node)

        else_node = graph.edge(do_node, NEXT)
        self.assertEdges(graph, else_node, {NEXT})

        finally_node = graph.edge(else_node, NEXT)
        self.assertEdges(graph, finally_node, {NEXT, RAISE})
        self.assertEdge(graph, finally_node, RAISE, context[RAISEC])
        self.assertEdge(graph, finally_node, NEXT, context[RETURN])

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
        graph, _, try_node = self._function_context(code)

        # The 'return' in the else branch should lead to the same place
        # as the handle_exception() success in the except branch.
        do_node = graph.edge(try_node, NEXT)
        raised_node = graph.edge(do_node, RAISE)
        ok_node = graph.edge(do_node, NEXT)

        raised_next = graph.edge(raised_node, NEXT)
        ok_next = graph.edge(ok_node, NEXT)

        self.assertEqual(raised_next, ok_next)

    def test_empty_module(self):
        code = ""
        graph, context, enter_node = self._module_context(code)
        self.assertEqual(enter_node, context[NEXTC])

    def test_just_pass(self):
        code = """\
pass
"""
        graph, context, enter_node = self._module_context(code)
        self.assertNodetype(enter_node, ast.Pass)
        self.assertEdges(graph, enter_node, {NEXT})
        self.assertEdge(graph, enter_node, NEXT, context[NEXTC])

    def test_statements_outside_function(self):
        code = """\
a = calculate()
try:
    something()
except:
    pass
"""
        graph, context, assign_node = self._module_context(code)
        self.assertNodetype(assign_node, ast.Assign)
        self.assertEdges(graph, assign_node, {NEXT, RAISE})
        self.assertEdge(graph, assign_node, RAISE, context[RAISEC])

        try_node = graph.edge(assign_node, NEXT)
        self.assertNodetype(try_node, ast.Try)
        self.assertEdges(graph, try_node, {NEXT})

        do_node = graph.edge(try_node, NEXT)
        self.assertNodetype(do_node, ast.Expr)
        self.assertEdges(graph, do_node, {NEXT, RAISE})
        self.assertEdge(graph, do_node, NEXT, context[NEXTC])

        pass_node = graph.edge(do_node, RAISE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(graph, pass_node, {NEXT})
        self.assertEdge(graph, pass_node, NEXT, context[NEXTC])

    def test_with(self):
        code = """\
with some_cm() as name:
    do_something()
"""
        graph, context, with_node = self._module_context(code)
        self.assertNodetype(with_node, ast.With)
        self.assertEdges(graph, with_node, {ENTER, RAISE})
        self.assertEdge(graph, with_node, RAISE, context[RAISEC])

        body_node = graph.edge(with_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(graph, body_node, {NEXT, RAISE})
        self.assertEdge(graph, body_node, RAISE, context[RAISEC])
        self.assertEdge(graph, body_node, NEXT, context[NEXTC])

    def test_async_for(self):
        code = """\
async def f():
    async for x in g():
        yield x*x
"""
        graph, context, for_node = self._function_context(code)
        self.assertNodetype(for_node, ast.AsyncFor)
        self.assertEdges(graph, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(graph, for_node, ELSE, context[RETURN])
        self.assertEdge(graph, for_node, RAISE, context[RAISEC])

        yield_node = graph.edge(for_node, ENTER)
        self.assertNodetype(yield_node, ast.Expr)
        self.assertEdges(graph, yield_node, {NEXT, RAISE})
        self.assertEdge(graph, yield_node, NEXT, for_node)
        self.assertEdge(graph, yield_node, RAISE, context[RAISEC])

    def test_async_with(self):
        code = """\
async def f():
    async with my_async_context():
        pass
"""
        graph, context, with_node = self._function_context(code)
        self.assertNodetype(with_node, ast.AsyncWith)
        self.assertEdges(graph, with_node, {ENTER, RAISE})
        self.assertEdge(graph, with_node, RAISE, context[RAISEC])

        pass_node = graph.edge(with_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(graph, pass_node, {NEXT})
        self.assertEdge(graph, pass_node, NEXT, context[RETURN])

    def test_classdef(self):
        code = """\
class SomeClass:
    def some_method(self, arg1, arg2):
        return bob
"""
        graph, context, initial = self._class_context(code)
        self.assertNodetype(initial, ast.FunctionDef)
        self.assertEdges(graph, initial, {NEXT, RAISE})
        self.assertEdge(graph, initial, NEXT, context[NEXTC])
        self.assertEdge(graph, initial, RAISE, context[RAISEC])

    def test_global(self):
        code = """\
def f():
    global bob
"""
        graph, context, node = self._function_context(code)
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
        node = context[ENTERC]
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
        graph, context, node = self._module_context(code)

        for _ in range(9):
            self.assertNodetype(node, ast.stmt)
            self.assertEdges(graph, node, {NEXT, RAISE})
            self.assertEdge(graph, node, RAISE, context[RAISEC])
            node = graph.edge(node, NEXT)

        self.assertEqual(node, context[NEXTC])

    # Assertions

    def assertEdges(self, graph, node, edges):
        """
        Assert that the outward edges from a node have the given names.
        """
        self.assertEqual(graph.edge_labels(node), edges)

    def assertEdge(self, graph, source, label, target):
        """
        Assert that the given edge from the given source maps to the
        given target.
        """
        self.assertEqual(graph.edge(source, label), target)

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
        self.assertEqual(
            sorted(context.keys()),
            [ENTERC, RAISEC, RETURN, RETURN_VALUE],
        )
        self.assertEdges(graph, context[RAISEC], set())
        self.assertEdges(graph, context[RETURN], set())
        self.assertEdges(graph, context[RETURN_VALUE], set())

        return graph, context, context[ENTERC]

    def _module_context(self, code):
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        self.assertIsInstance(module_node, ast.Module)

        graph = CFAnalysis.from_module(module_node)
        context = graph.context
        self.assertEqual(
            sorted(context.keys()),
            [ENTERC, NEXTC, RAISEC],
        )
        self.assertEdges(graph, context[RAISEC], set())
        self.assertEdges(graph, context[NEXTC], set())

        return graph, context, context[ENTERC]

    def _class_context(self, code):
        (module_node,) = compile(
            code, "test_cf", "exec", ast.PyCF_ONLY_AST
        ).body
        self.assertIsInstance(module_node, ast.ClassDef)

        graph = CFAnalysis.from_class(module_node)
        context = graph.context
        self.assertEqual(
            sorted(context.keys()),
            [ENTERC, NEXTC, RAISEC],
        )
        self.assertEdges(graph, context[RAISEC], set())
        self.assertEdges(graph, context[NEXTC], set())

        return graph, context, context[ENTERC]
