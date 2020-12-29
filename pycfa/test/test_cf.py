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
Tests for CFAnalyser class.
"""

import ast
import unittest

from pycfa.cf import (
    CFAnalyser,
    ELSE,
    ENTER,
    NEXT,
    RAISE,
)


class TestCFAnalyser(unittest.TestCase):
    def test_analyse_noop_function(self):
        code = """\
def f():
    pass
"""
        analysis, pass_node = self._function_context(code)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.return_without_value)

    def test_analyse_single_expr_statement(self):
        code = """\
def f():
    do_something()
"""
        analysis, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Expr)
        self.assertEdges(analysis, stmt_node, {NEXT, RAISE})
        self.assertEdge(analysis, stmt_node, NEXT, analysis.return_without_value)
        self.assertEdge(analysis, stmt_node, RAISE, analysis.raise_node)

    def test_analyse_assign(self):
        code = """\
def f():
    a = 123
"""
        analysis, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Assign)
        self.assertEdges(analysis, stmt_node, {NEXT, RAISE})
        self.assertEdge(analysis, stmt_node, NEXT, analysis.return_without_value)
        self.assertEdge(analysis, stmt_node, RAISE, analysis.raise_node)

    def test_analyse_multiple_statements(self):
        code = """\
def f():
    do_something()
    a += do_something_else()
"""
        analysis, stmt1_node = self._function_context(code)
        self.assertNodetype(stmt1_node, ast.Expr)
        self.assertEdges(analysis, stmt1_node, {NEXT, RAISE})
        self.assertEdge(analysis, stmt1_node, RAISE, analysis.raise_node)

        stmt2_node = analysis.edge(stmt1_node, NEXT)
        self.assertNodetype(stmt2_node, ast.AugAssign)
        self.assertEdges(analysis, stmt2_node, {NEXT, RAISE})
        self.assertEdge(analysis, stmt2_node, NEXT, analysis.return_without_value)
        self.assertEdge(analysis, stmt2_node, RAISE, analysis.raise_node)

    def test_return_with_no_value(self):
        code = """\
def f():
    return
"""
        analysis, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Return)
        self.assertEdges(analysis, stmt_node, {NEXT})
        self.assertEdge(analysis, stmt_node, NEXT, analysis.return_without_value)

    def test_return_with_value(self):
        code = """\
def f():
    return None
"""
        analysis, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Return)
        self.assertEdges(analysis, stmt_node, {NEXT, RAISE})
        self.assertEdge(analysis, stmt_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, stmt_node, NEXT, analysis.return_with_value)

    def test_raise(self):
        code = """\
def f():
    raise TypeError("don't call me")
"""
        analysis, stmt_node = self._function_context(code)
        self.assertNodetype(stmt_node, ast.Raise)
        self.assertEdges(analysis, stmt_node, {RAISE})
        self.assertEdge(analysis, stmt_node, RAISE, analysis.raise_node)

    def test_if(self):
        code = """\
def f():
    if condition:
        a = 123
"""
        analysis, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, if_node, RAISE, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(analysis, if_branch, {NEXT, RAISE})
        self.assertEdge(analysis, if_branch, RAISE, analysis.raise_node)
        self.assertEdge(analysis, if_branch, NEXT, analysis.return_without_value)
        self.assertEdge(analysis, if_node, ELSE, analysis.return_without_value)

    def test_if_else(self):
        code = """\
def f():
    if condition:
        a = 123
    else:
        b = 456
"""
        analysis, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, if_node, RAISE, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(analysis, if_branch, {NEXT, RAISE})
        self.assertEdge(analysis, if_branch, RAISE, analysis.raise_node)
        self.assertEdge(analysis, if_branch, NEXT, analysis.return_without_value)

        else_branch = analysis.edge(if_node, ELSE)
        self.assertNodetype(else_branch, ast.Assign)
        self.assertEdges(analysis, else_branch, {NEXT, RAISE})
        self.assertEdge(analysis, else_branch, RAISE, analysis.raise_node)
        self.assertEdge(analysis, else_branch, NEXT, analysis.return_without_value)

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
        analysis, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, if_node, RAISE, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(analysis, if_branch, {NEXT, RAISE})
        self.assertEdge(analysis, if_branch, RAISE, analysis.raise_node)
        self.assertEdge(analysis, if_branch, NEXT, analysis.return_without_value)

        elif_node = analysis.edge(if_node, ELSE)
        self.assertNodetype(elif_node, ast.If)
        self.assertEdges(analysis, elif_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, elif_node, RAISE, analysis.raise_node)

        elif_branch = analysis.edge(elif_node, ENTER)
        self.assertNodetype(elif_branch, ast.Assign)
        self.assertEdges(analysis, elif_branch, {NEXT, RAISE})
        self.assertEdge(analysis, elif_branch, RAISE, analysis.raise_node)
        self.assertEdge(analysis, elif_branch, NEXT, analysis.return_without_value)

        else_branch = analysis.edge(elif_node, ELSE)
        self.assertNodetype(else_branch, ast.Assign)
        self.assertEdges(analysis, else_branch, {NEXT, RAISE})
        self.assertEdge(analysis, else_branch, RAISE, analysis.raise_node)
        self.assertEdge(analysis, else_branch, NEXT, analysis.return_without_value)

    def test_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return 123
    else:
        return 456
"""
        analysis, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, if_node, RAISE, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Return)
        self.assertEdges(analysis, if_branch, {NEXT, RAISE})
        self.assertEdge(analysis, if_branch, NEXT, analysis.return_with_value)
        self.assertEdge(analysis, if_branch, RAISE, analysis.raise_node)

        else_node = analysis.edge(if_node, ELSE)
        self.assertNodetype(else_node, ast.Return)
        self.assertEdges(analysis, else_node, {NEXT, RAISE})
        self.assertEdge(analysis, else_node, NEXT, analysis.return_with_value)
        self.assertEdge(analysis, else_node, RAISE, analysis.raise_node)

    def test_plain_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return
    else:
        return
"""
        analysis, if_node = self._function_context(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, if_node, RAISE, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Return)
        self.assertEdges(analysis, if_branch, {NEXT})
        self.assertEdge(analysis, if_branch, NEXT, analysis.return_without_value)

        else_node = analysis.edge(if_node, ELSE)
        self.assertNodetype(else_node, ast.Return)
        self.assertEdges(analysis, else_node, {NEXT})
        self.assertEdge(analysis, else_node, NEXT, analysis.return_without_value)

    def test_unreachable_statements(self):
        code = """\
def f():
    do_something()
    return
    do_something_else()
"""
        analysis, stmt1_node = self._function_context(code)
        self.assertNodetype(stmt1_node, ast.Expr)
        self.assertEdges(analysis, stmt1_node, {NEXT, RAISE})
        self.assertEdge(analysis, stmt1_node, RAISE, analysis.raise_node)

        stmt2_node = analysis.edge(stmt1_node, NEXT)
        self.assertNodetype(stmt2_node, ast.Return)
        self.assertEdges(analysis, stmt2_node, {NEXT})
        self.assertEdge(analysis, stmt2_node, NEXT, analysis.return_without_value)

    def test_while(self):
        code = """\
def f():
    while some_condition:
        do_something()
"""
        analysis, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, while_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, while_node, ELSE, analysis.return_without_value)

        body_node = analysis.edge(while_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, RAISE})
        self.assertEdge(analysis, body_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, while_node)

    def test_while_else(self):
        code = """\
def f():
    while some_condition:
        do_something()
    else:
        do_no_break_stuff()
"""
        analysis, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, while_node, RAISE, analysis.raise_node)

        body_node = analysis.edge(while_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, RAISE})
        self.assertEdge(analysis, body_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, while_node)

        else_node = analysis.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, RAISE})
        self.assertEdge(analysis, else_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.return_without_value)

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
        analysis, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, while_node, RAISE, analysis.raise_node)

        test_node = analysis.edge(while_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(analysis, test_node, RAISE, analysis.raise_node)
        self.assertEdges(analysis, test_node, {ENTER, ELSE, RAISE})

        continue_node = analysis.edge(test_node, ENTER)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(analysis, continue_node, {NEXT})
        self.assertEdge(analysis, continue_node, NEXT, while_node)

        body_node = analysis.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, RAISE})
        self.assertEdge(analysis, body_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, while_node)

        else_node = analysis.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, RAISE})
        self.assertEdge(analysis, else_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.return_without_value)

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
        analysis, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, while_node, RAISE, analysis.raise_node)

        test_node = analysis.edge(while_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(analysis, test_node, RAISE, analysis.raise_node)
        self.assertEdges(analysis, test_node, {ENTER, ELSE, RAISE})

        break_node = analysis.edge(test_node, ENTER)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(analysis, break_node, {NEXT})
        self.assertEdge(analysis, break_node, NEXT, analysis.return_without_value)

        body_node = analysis.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, RAISE})
        self.assertEdge(analysis, body_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, while_node)

        else_node = analysis.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, RAISE})
        self.assertEdge(analysis, else_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.return_without_value)

    def test_while_with_two_statements(self):
        code = """\
def f():
    while some_condition:
        do_something()
        do_something_else()
"""
        analysis, while_node = self._function_context(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, while_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, while_node, ELSE, analysis.return_without_value)

        body_node1 = analysis.edge(while_node, ENTER)
        self.assertNodetype(body_node1, ast.Expr)
        self.assertEdges(analysis, body_node1, {NEXT, RAISE})
        self.assertEdge(analysis, body_node1, RAISE, analysis.raise_node)

        body_node2 = analysis.edge(body_node1, NEXT)
        self.assertNodetype(body_node2, ast.Expr)
        self.assertEdges(analysis, body_node2, {NEXT, RAISE})
        self.assertEdge(analysis, body_node2, RAISE, analysis.raise_node)
        self.assertEdge(analysis, body_node2, NEXT, while_node)

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
        analysis, for_node = self._function_context(code)
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        test_node = analysis.edge(for_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(analysis, test_node, RAISE, analysis.raise_node)
        self.assertEdges(analysis, test_node, {ENTER, ELSE, RAISE})

        continue_node = analysis.edge(test_node, ENTER)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(analysis, continue_node, {NEXT})
        self.assertEdge(analysis, continue_node, NEXT, for_node)

        body_node = analysis.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, RAISE})
        self.assertEdge(analysis, body_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, for_node)

        else_node = analysis.edge(for_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, RAISE})
        self.assertEdge(analysis, else_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.return_without_value)

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
        analysis, for_node = self._function_context(code)
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        test_node = analysis.edge(for_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(analysis, test_node, RAISE, analysis.raise_node)
        self.assertEdges(analysis, test_node, {ENTER, ELSE, RAISE})

        break_node = analysis.edge(test_node, ENTER)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(analysis, break_node, {NEXT})
        self.assertEdge(analysis, break_node, NEXT, analysis.return_without_value)

        body_node = analysis.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, RAISE})
        self.assertEdge(analysis, body_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, for_node)

        else_node = analysis.edge(for_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, RAISE})
        self.assertEdge(analysis, else_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.return_without_value)

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
        analysis, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Expr)
        self.assertEdges(analysis, try_node, {NEXT, RAISE})

        except1_node = analysis.edge(try_node, RAISE)
        self.assertNodetype(except1_node, ast.expr)
        self.assertEdges(analysis, except1_node, {ENTER, ELSE, RAISE})
        self.assertEdge(analysis, except1_node, RAISE, analysis.raise_node)

        match1_node = analysis.edge(except1_node, ENTER)
        self.assertNodetype(match1_node, ast.Expr)
        self.assertEdges(analysis, match1_node, {NEXT, RAISE})
        self.assertEdge(analysis, match1_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, match1_node, NEXT, analysis.return_without_value)

        match2_node = analysis.edge(except1_node, ELSE)
        self.assertNodetype(match2_node, ast.Expr)
        self.assertEdges(analysis, match2_node, {NEXT, RAISE})
        self.assertEdge(analysis, match2_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, match2_node, NEXT, analysis.return_without_value)

        else_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, RAISE})
        self.assertEdge(analysis, else_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.return_without_value)

    def test_try_except_pass(self):
        code = """\
def f():
    try:
        something_dangerous()
    except:
        pass
"""
        analysis, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Expr)
        self.assertEdges(analysis, try_node, {NEXT, RAISE})
        self.assertEdge(analysis, try_node, NEXT, analysis.return_without_value)

        pass_node = analysis.edge(try_node, RAISE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.return_without_value)

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
        analysis, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Raise)
        self.assertEdges(analysis, try_node, {RAISE})

        except_node = analysis.edge(try_node, RAISE)
        self.assertNodetype(except_node, ast.expr)
        self.assertEdges(analysis, except_node, {ENTER, ELSE, RAISE})
        self.assertEdge(analysis, except_node, ELSE, analysis.raise_node)
        self.assertEdge(analysis, except_node, RAISE, analysis.raise_node)

        pass_node = analysis.edge(except_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.return_without_value)

    def test_try_finally_pass(self):
        code = """\
def f():
    try:
        pass
    finally:
        do_something()
"""
        analysis, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Pass)
        self.assertEdges(analysis, try_node, {NEXT})

        finally_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.return_without_value)

    def test_try_finally_raise(self):
        code = """\
def f():
    try:
        raise ValueError()
    finally:
        do_something()
"""
        analysis, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Raise)
        self.assertEdges(analysis, try_node, {RAISE})

        finally_node = analysis.edge(try_node, RAISE)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.raise_node)

    def test_try_finally_return(self):
        code = """\
def f():
    try:
        return
    finally:
        do_something()
"""
        analysis, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Return)
        self.assertEdges(analysis, try_node, {NEXT})

        finally_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.return_without_value)

    def test_try_finally_return_value(self):
        code = """\
def f():
    try:
        return "abc"
    finally:
        do_something()
"""
        analysis, start_node = self._function_context(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Return)
        self.assertEdges(analysis, try_node, {NEXT, RAISE})

        finally_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.return_with_value)

        finally2_node = analysis.edge(try_node, RAISE)
        self.assertNodetype(finally2_node, ast.Expr)
        self.assertEdges(analysis, finally2_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally2_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally2_node, NEXT, analysis.raise_node)

    def test_try_finally_break(self):
        code = """\
def f():
    for item in items:
        try:
            break
        finally:
            do_something()
"""
        analysis, for_node = self._function_context(code)
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertNodetype(try_node, ast.Try)
        self.assertEdges(analysis, try_node, {NEXT})

        break_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(analysis, break_node, {NEXT})

        finally_node = analysis.edge(break_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.return_without_value)

    def test_try_finally_continue(self):
        code = """\
def f():
    for item in items:
        try:
            continue
        finally:
            do_something()
"""
        analysis, for_node = self._function_context(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        continue_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(analysis, continue_node, {NEXT})

        finally_node = analysis.edge(continue_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, for_node)

    def test_return_value_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return some_value()
"""
        analysis, try_node = self._function_context(code)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {RAISE})

        return_node = analysis.edge(raise_node, RAISE)
        self.assertEdges(analysis, return_node, {NEXT, RAISE})
        self.assertEdge(analysis, return_node, NEXT, analysis.return_with_value)
        self.assertEdge(analysis, return_node, RAISE, analysis.raise_node)

    def test_return_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return
"""
        analysis, try_node = self._function_context(code)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {RAISE})

        return_node = analysis.edge(raise_node, RAISE)
        self.assertEdges(analysis, return_node, {NEXT})
        self.assertEdge(analysis, return_node, NEXT, analysis.return_without_value)

    def test_raise_in_finally(self):
        code = """\
def f():
    try:
        pass
    finally:
        raise SomeException()
"""
        analysis, try_node = self._function_context(code)
        self.assertEdges(analysis, try_node, {NEXT})

        pass_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})

        raise_node = analysis.edge(pass_node, NEXT)
        self.assertNodetype(raise_node, ast.Raise)
        self.assertEdges(analysis, raise_node, {RAISE})
        self.assertEdge(analysis, raise_node, RAISE, analysis.raise_node)

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
        analysis, for_node = self._function_context(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        return_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, return_node, {NEXT})

        break_node = analysis.edge(return_node, NEXT)
        self.assertEdges(analysis, break_node, {NEXT})
        self.assertEdge(analysis, break_node, NEXT, analysis.return_without_value)

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
        analysis, for_node = self._function_context(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {RAISE})

        continue_node = analysis.edge(raise_node, RAISE)
        self.assertEdges(analysis, continue_node, {NEXT})
        self.assertEdge(analysis, continue_node, NEXT, for_node)

    def test_continue_in_except_no_finally(self):
        code = """\
def f():
    for item in item_factory():
        try:
            raise SomeException()
        except:
            continue
"""
        analysis, for_node = self._function_context(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {RAISE})

        continue_node = analysis.edge(raise_node, RAISE)
        self.assertEdges(analysis, continue_node, {NEXT})
        self.assertEdge(analysis, continue_node, NEXT, for_node)

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
        analysis, for_node = self._function_context(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {RAISE})

        continue_node = analysis.edge(raise_node, RAISE)
        self.assertEdges(analysis, continue_node, {NEXT})

        finally_node = analysis.edge(continue_node, NEXT)
        self.assertEdges(analysis, finally_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, for_node)

    def test_break_in_inner_loop(self):
        code = """\
def f():
    while some_condition():
        while some_other_condition():
            break
"""
        analysis, while_node = self._function_context(code)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, while_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, while_node, RAISE, analysis.raise_node)

        inner_while_node = analysis.edge(while_node, ENTER)
        self.assertEdges(analysis, inner_while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, inner_while_node, ELSE, while_node)
        self.assertEdge(analysis, inner_while_node, RAISE, analysis.raise_node)

        break_node = analysis.edge(inner_while_node, ENTER)
        self.assertEdges(analysis, break_node, {NEXT})
        self.assertEdge(analysis, break_node, NEXT, while_node)

    def test_continue_in_inner_loop(self):
        code = """\
def f():
    while some_condition():
        while some_other_condition():
            continue
"""
        analysis, while_node = self._function_context(code)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, while_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, while_node, RAISE, analysis.raise_node)

        inner_while_node = analysis.edge(while_node, ENTER)
        self.assertEdges(analysis, inner_while_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, inner_while_node, ELSE, while_node)
        self.assertEdge(analysis, inner_while_node, RAISE, analysis.raise_node)

        continue_node = analysis.edge(inner_while_node, ENTER)
        self.assertEdges(analysis, continue_node, {NEXT})
        self.assertEdge(analysis, continue_node, NEXT, inner_while_node)

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
        analysis, for_node = self._function_context(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {RAISE})

        except_node = analysis.edge(raise_node, RAISE)
        self.assertEdges(analysis, except_node, {ENTER, ELSE, RAISE})
        self.assertEdge(analysis, except_node, RAISE, analysis.edge(except_node, ELSE))

        finally_raise_node = analysis.edge(except_node, RAISE)
        self.assertEdges(analysis, finally_raise_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_raise_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_raise_node, NEXT, analysis.raise_node)

        break_node = analysis.edge(except_node, ENTER)
        self.assertEdges(analysis, break_node, {NEXT})

        finally_node = analysis.edge(break_node, NEXT)
        self.assertEdges(analysis, finally_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.return_without_value)

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
        analysis, for_node = self._function_context(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        do_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, do_node, {NEXT, RAISE})

        pass_node = analysis.edge(do_node, RAISE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})

        finally1_node = analysis.edge(pass_node, NEXT)
        self.assertEdges(analysis, finally1_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally1_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally1_node, NEXT, for_node)

        else_node = analysis.edge(do_node, NEXT)
        self.assertEdges(analysis, else_node, {NEXT})

        finally_node = analysis.edge(else_node, NEXT)
        self.assertEdges(analysis, finally_node, {NEXT, RAISE})
        self.assertEdge(analysis, finally_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.return_without_value)

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
        analysis, try_node = self._function_context(code)

        # The 'return' in the else branch should lead to the same place
        # as the handle_exception() success in the except branch.
        do_node = analysis.edge(try_node, NEXT)
        raised_node = analysis.edge(do_node, RAISE)
        ok_node = analysis.edge(do_node, NEXT)

        raised_next = analysis.edge(raised_node, NEXT)
        ok_next = analysis.edge(ok_node, NEXT)

        self.assertEqual(raised_next, ok_next)

    def test_empty_module(self):
        code = ""
        analysis, enter_node = self._module_context(code)
        self.assertEqual(enter_node, analysis.leave_node)

    def test_just_pass(self):
        code = """\
pass
"""
        analysis, enter_node = self._module_context(code)
        self.assertNodetype(enter_node, ast.Pass)
        self.assertEdges(analysis, enter_node, {NEXT})
        self.assertEdge(analysis, enter_node, NEXT, analysis.leave_node)

    def test_statements_outside_function(self):
        code = """\
a = calculate()
try:
    something()
except:
    pass
"""
        analysis, assign_node = self._module_context(code)
        self.assertNodetype(assign_node, ast.Assign)
        self.assertEdges(analysis, assign_node, {NEXT, RAISE})
        self.assertEdge(analysis, assign_node, RAISE, analysis.raise_node)

        try_node = analysis.edge(assign_node, NEXT)
        self.assertNodetype(try_node, ast.Try)
        self.assertEdges(analysis, try_node, {NEXT})

        do_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(do_node, ast.Expr)
        self.assertEdges(analysis, do_node, {NEXT, RAISE})
        self.assertEdge(analysis, do_node, NEXT, analysis.leave_node)

        pass_node = analysis.edge(do_node, RAISE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.leave_node)

    def test_with(self):
        code = """\
with some_cm() as name:
    do_something()
"""
        analysis, with_node = self._module_context(code)
        self.assertNodetype(with_node, ast.With)
        self.assertEdges(analysis, with_node, {ENTER, RAISE})
        self.assertEdge(analysis, with_node, RAISE, analysis.raise_node)

        body_node = analysis.edge(with_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, RAISE})
        self.assertEdge(analysis, body_node, RAISE, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, analysis.leave_node)

    def test_async_for(self):
        code = """\
async def f():
    async for x in g():
        yield x*x
"""
        analysis, for_node = self._function_context(code)
        self.assertNodetype(for_node, ast.AsyncFor)
        self.assertEdges(analysis, for_node, {ELSE, ENTER, RAISE})
        self.assertEdge(analysis, for_node, ELSE, analysis.return_without_value)
        self.assertEdge(analysis, for_node, RAISE, analysis.raise_node)

        yield_node = analysis.edge(for_node, ENTER)
        self.assertNodetype(yield_node, ast.Expr)
        self.assertEdges(analysis, yield_node, {NEXT, RAISE})
        self.assertEdge(analysis, yield_node, NEXT, for_node)
        self.assertEdge(analysis, yield_node, RAISE, analysis.raise_node)

    def test_async_with(self):
        code = """\
async def f():
    async with my_async_context():
        pass
"""
        analysis, with_node = self._function_context(code)
        self.assertNodetype(with_node, ast.AsyncWith)
        self.assertEdges(analysis, with_node, {ENTER, RAISE})
        self.assertEdge(analysis, with_node, RAISE, analysis.raise_node)

        pass_node = analysis.edge(with_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.return_without_value)

    def test_classdef(self):
        code = """\
class SomeClass:
    def some_method(self, arg1, arg2):
        return bob
"""
        analysis, initial = self._class_context(code)
        self.assertNodetype(initial, ast.FunctionDef)
        self.assertEdges(analysis, initial, {NEXT, RAISE})
        self.assertEdge(analysis, initial, NEXT, analysis.leave_node)
        self.assertEdge(analysis, initial, RAISE, analysis.raise_node)

    def test_global(self):
        code = """\
def f():
    global bob
"""
        analysis, _ = self._function_context(code)
        node = analysis.entry_node
        self.assertNodetype(node, ast.Global)
        self.assertEdges(analysis, node, {NEXT})
        self.assertEdge(analysis, node, NEXT, analysis.return_without_value)

    def test_nonlocal(self):
        code = """\
def f(bob):
    def g():
        nonlocal bob
"""
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        (function_node,) = module_node.body
        (inner_function,) = function_node.body

        analysis = CFAnalyser().analyse_function(inner_function)
        node = analysis.entry_node
        self.assertNodetype(node, ast.Nonlocal)
        self.assertEdges(analysis, node, {NEXT})
        self.assertEdge(analysis, node, NEXT, analysis.return_without_value)

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
        analysis, node = self._module_context(code)

        for _ in range(9):
            self.assertNodetype(node, ast.stmt)
            self.assertEdges(analysis, node, {NEXT, RAISE})
            self.assertEdge(analysis, node, RAISE, analysis.raise_node)
            node = analysis.edge(node, NEXT)

        self.assertEqual(node, analysis.leave_node)

    def test_function_cant_raise(self):
        code = """\
def f():
    try:
        something_or_other()
    except:
        pass
"""
        analysis, _ = self._function_context(code)
        with self.assertRaises(AttributeError):
            analysis.raise_node

    def test_function_no_return_value(self):
        code = """\
def f():
    pass
"""
        analysis, _ = self._function_context(code)
        with self.assertRaises(AttributeError):
            analysis.return_with_value

    def test_function_no_return_without_value(self):
        code = """\
def f():
    return 123
"""
        analysis, _ = self._function_context(code)
        with self.assertRaises(AttributeError):
            analysis.return_without_value

    def test_class_cant_raise(self):
        code = """\
class A:
    try:
        something_or_other()
    except:
        pass
"""
        analysis, _ = self._class_context(code)
        with self.assertRaises(AttributeError):
            analysis.raise_node

    def test_module_cant_raise(self):
        code = """\
try:
    something_or_other()
except:
    pass
"""
        analysis, _ = self._module_context(code)
        with self.assertRaises(AttributeError):
            analysis.raise_node

    # Assertions

    def assertEdges(self, analysis, node, edges):
        """
        Assert that the outward edges from a node have the given names.
        """
        self.assertEqual(analysis.edge_labels(node), edges)

    def assertEdge(self, analysis, source, label, target):
        """
        Assert that the given edge from the given source maps to the
        given target.
        """
        self.assertEqual(analysis.edge(source, label), target)

    def assertNodetype(self, node, nodetype):
        """
        Assert that the given control-flow analysis node is associated
        to an ast node of the given type.
        """
        self.assertIsInstance(node.ast_node, nodetype)

    # Helper methods

    def _function_context(self, code):
        (function_node,) = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST).body
        self.assertIsInstance(function_node, (ast.AsyncFunctionDef, ast.FunctionDef))

        analysis = CFAnalyser().analyse_function(function_node)
        if hasattr(analysis, "raise_node"):
            self.assertEdges(analysis, analysis.raise_node, set())
        if hasattr(analysis, "return_with_value"):
            self.assertEdges(analysis, analysis.return_with_value, set())
        if hasattr(analysis, "return_without_value"):
            self.assertEdges(analysis, analysis.return_without_value, set())

        return analysis, analysis.entry_node

    def _module_context(self, code):
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        self.assertIsInstance(module_node, ast.Module)

        analysis = CFAnalyser().analyse_module(module_node)

        if hasattr(analysis, "raise_node"):
            self.assertEdges(analysis, analysis.raise_node, set())
        if hasattr(analysis, "leave_node"):
            self.assertEdges(analysis, analysis.leave_node, set())

        return analysis, analysis.entry_node

    def _class_context(self, code):
        (module_node,) = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST).body
        self.assertIsInstance(module_node, ast.ClassDef)

        analysis = CFAnalyser().analyse_class(module_node)

        if hasattr(analysis, "raise_node"):
            self.assertEdges(analysis, analysis.raise_node, set())
        if hasattr(analysis, "leave_node"):
            self.assertEdges(analysis, analysis.leave_node, set())

        return analysis, analysis.entry_node
