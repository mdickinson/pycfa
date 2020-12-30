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
from typing import Set

from pycfa.cfanalyser import CFAnalyser, ELSE, ENTER, ERROR, NEXT
from pycfa.cfanalysis import CFAnalysis


def all_statements(tree: ast.AST) -> Set[ast.stmt]:
    """
    Return the set of all ast.stmt nodes in a tree.
    """
    return {node for node in ast.walk(tree) if isinstance(node, ast.stmt)}


def analysed_statements(analysis: CFAnalysis) -> Set[ast.stmt]:
    """
    Return the set of all ast statements that have a corresponding node
    in an analysis object.
    """
    return {
        node.ast_node
        for node in analysis.nodes()
        if hasattr(node, "ast_node")
        if isinstance(node.ast_node, ast.stmt)
    }


class TestCFAnalyser(unittest.TestCase):
    def test_analyse_noop_function(self):
        code = """\
def f():
    pass
"""
        analysis, pass_node = self._function_analysis(code)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.leave_node)

    def test_analyse_single_expr_statement(self):
        code = """\
def f():
    do_something()
"""
        analysis, stmt_node = self._function_analysis(code)
        self.assertNodetype(stmt_node, ast.Expr)
        self.assertEdges(analysis, stmt_node, {NEXT, ERROR})
        self.assertEdge(analysis, stmt_node, NEXT, analysis.leave_node)
        self.assertEdge(analysis, stmt_node, ERROR, analysis.raise_node)

    def test_analyse_assign(self):
        code = """\
def f():
    a = 123
"""
        analysis, stmt_node = self._function_analysis(code)
        self.assertNodetype(stmt_node, ast.Assign)
        self.assertEdges(analysis, stmt_node, {NEXT, ERROR})
        self.assertEdge(analysis, stmt_node, NEXT, analysis.leave_node)
        self.assertEdge(analysis, stmt_node, ERROR, analysis.raise_node)

    def test_analyse_multiple_statements(self):
        code = """\
def f():
    do_something()
    a += do_something_else()
"""
        analysis, stmt1_node = self._function_analysis(code)
        self.assertNodetype(stmt1_node, ast.Expr)
        self.assertEdges(analysis, stmt1_node, {NEXT, ERROR})
        self.assertEdge(analysis, stmt1_node, ERROR, analysis.raise_node)

        stmt2_node = analysis.edge(stmt1_node, NEXT)
        self.assertNodetype(stmt2_node, ast.AugAssign)
        self.assertEdges(analysis, stmt2_node, {NEXT, ERROR})
        self.assertEdge(analysis, stmt2_node, NEXT, analysis.leave_node)
        self.assertEdge(analysis, stmt2_node, ERROR, analysis.raise_node)

    def test_return_with_no_value(self):
        code = """\
def f():
    return
"""
        analysis, stmt_node = self._function_analysis(code)
        self.assertNodetype(stmt_node, ast.Return)
        self.assertEdges(analysis, stmt_node, {NEXT})
        self.assertEdge(analysis, stmt_node, NEXT, analysis.leave_node)

    def test_return_with_value(self):
        code = """\
def f():
    return None
"""
        analysis, stmt_node = self._function_analysis(code)
        self.assertNodetype(stmt_node, ast.Return)
        self.assertEdges(analysis, stmt_node, {NEXT, ERROR})
        self.assertEdge(analysis, stmt_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, stmt_node, NEXT, analysis.return_node)

    def test_raise(self):
        code = """\
def f():
    raise TypeError("don't call me")
"""
        analysis, stmt_node = self._function_analysis(code)
        self.assertNodetype(stmt_node, ast.Raise)
        self.assertEdges(analysis, stmt_node, {ERROR})
        self.assertEdge(analysis, stmt_node, ERROR, analysis.raise_node)

    def test_if(self):
        code = """\
def f():
    if condition:
        a = 123
"""
        analysis, if_node = self._function_analysis(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, if_node, ERROR, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(analysis, if_branch, {NEXT, ERROR})
        self.assertEdge(analysis, if_branch, ERROR, analysis.raise_node)
        self.assertEdge(analysis, if_branch, NEXT, analysis.leave_node)
        self.assertEdge(analysis, if_node, ELSE, analysis.leave_node)

    def test_if_else(self):
        code = """\
def f():
    if condition:
        a = 123
    else:
        b = 456
"""
        analysis, if_node = self._function_analysis(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, if_node, ERROR, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(analysis, if_branch, {NEXT, ERROR})
        self.assertEdge(analysis, if_branch, ERROR, analysis.raise_node)
        self.assertEdge(analysis, if_branch, NEXT, analysis.leave_node)

        else_branch = analysis.edge(if_node, ELSE)
        self.assertNodetype(else_branch, ast.Assign)
        self.assertEdges(analysis, else_branch, {NEXT, ERROR})
        self.assertEdge(analysis, else_branch, ERROR, analysis.raise_node)
        self.assertEdge(analysis, else_branch, NEXT, analysis.leave_node)

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
        analysis, if_node = self._function_analysis(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, if_node, ERROR, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Assign)
        self.assertEdges(analysis, if_branch, {NEXT, ERROR})
        self.assertEdge(analysis, if_branch, ERROR, analysis.raise_node)
        self.assertEdge(analysis, if_branch, NEXT, analysis.leave_node)

        elif_node = analysis.edge(if_node, ELSE)
        self.assertNodetype(elif_node, ast.If)
        self.assertEdges(analysis, elif_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, elif_node, ERROR, analysis.raise_node)

        elif_branch = analysis.edge(elif_node, ENTER)
        self.assertNodetype(elif_branch, ast.Assign)
        self.assertEdges(analysis, elif_branch, {NEXT, ERROR})
        self.assertEdge(analysis, elif_branch, ERROR, analysis.raise_node)
        self.assertEdge(analysis, elif_branch, NEXT, analysis.leave_node)

        else_branch = analysis.edge(elif_node, ELSE)
        self.assertNodetype(else_branch, ast.Assign)
        self.assertEdges(analysis, else_branch, {NEXT, ERROR})
        self.assertEdge(analysis, else_branch, ERROR, analysis.raise_node)
        self.assertEdge(analysis, else_branch, NEXT, analysis.leave_node)

    def test_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return 123
    else:
        return 456
"""
        analysis, if_node = self._function_analysis(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, if_node, ERROR, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Return)
        self.assertEdges(analysis, if_branch, {NEXT, ERROR})
        self.assertEdge(analysis, if_branch, NEXT, analysis.return_node)
        self.assertEdge(analysis, if_branch, ERROR, analysis.raise_node)

        else_node = analysis.edge(if_node, ELSE)
        self.assertNodetype(else_node, ast.Return)
        self.assertEdges(analysis, else_node, {NEXT, ERROR})
        self.assertEdge(analysis, else_node, NEXT, analysis.return_node)
        self.assertEdge(analysis, else_node, ERROR, analysis.raise_node)

    def test_plain_return_in_if_and_else(self):
        code = """\
def f():
    if condition:
        return
    else:
        return
"""
        analysis, if_node = self._function_analysis(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, if_node, ERROR, analysis.raise_node)

        if_branch = analysis.edge(if_node, ENTER)
        self.assertNodetype(if_branch, ast.Return)
        self.assertEdges(analysis, if_branch, {NEXT})
        self.assertEdge(analysis, if_branch, NEXT, analysis.leave_node)

        else_node = analysis.edge(if_node, ELSE)
        self.assertNodetype(else_node, ast.Return)
        self.assertEdges(analysis, else_node, {NEXT})
        self.assertEdge(analysis, else_node, NEXT, analysis.leave_node)

    def test_unreachable_statements(self):
        code = """\
def f():
    do_something()
    return
    do_something_else()
"""
        analysis, stmt1_node = self._function_analysis(code)
        self.assertNodetype(stmt1_node, ast.Expr)
        self.assertEdges(analysis, stmt1_node, {NEXT, ERROR})
        self.assertEdge(analysis, stmt1_node, ERROR, analysis.raise_node)

        stmt2_node = analysis.edge(stmt1_node, NEXT)
        self.assertNodetype(stmt2_node, ast.Return)
        self.assertEdges(analysis, stmt2_node, {NEXT})
        self.assertEdge(analysis, stmt2_node, NEXT, analysis.leave_node)

    def test_while(self):
        code = """\
def f():
    while some_condition:
        do_something()
"""
        analysis, while_node = self._function_analysis(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, while_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, while_node, ELSE, analysis.leave_node)

        body_node = analysis.edge(while_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, ERROR})
        self.assertEdge(analysis, body_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, while_node)

    def test_while_else(self):
        code = """\
def f():
    while some_condition:
        do_something()
    else:
        do_no_break_stuff()
"""
        analysis, while_node = self._function_analysis(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, while_node, ERROR, analysis.raise_node)

        body_node = analysis.edge(while_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, ERROR})
        self.assertEdge(analysis, body_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, while_node)

        else_node = analysis.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, ERROR})
        self.assertEdge(analysis, else_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.leave_node)

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
        analysis, while_node = self._function_analysis(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, while_node, ERROR, analysis.raise_node)

        test_node = analysis.edge(while_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(analysis, test_node, ERROR, analysis.raise_node)
        self.assertEdges(analysis, test_node, {ENTER, ELSE, ERROR})

        continue_node = analysis.edge(test_node, ENTER)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(analysis, continue_node, {NEXT})
        self.assertEdge(analysis, continue_node, NEXT, while_node)

        body_node = analysis.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, ERROR})
        self.assertEdge(analysis, body_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, while_node)

        else_node = analysis.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, ERROR})
        self.assertEdge(analysis, else_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.leave_node)

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
        analysis, while_node = self._function_analysis(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, while_node, ERROR, analysis.raise_node)

        test_node = analysis.edge(while_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(analysis, test_node, ERROR, analysis.raise_node)
        self.assertEdges(analysis, test_node, {ENTER, ELSE, ERROR})

        break_node = analysis.edge(test_node, ENTER)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(analysis, break_node, {NEXT})
        self.assertEdge(analysis, break_node, NEXT, analysis.leave_node)

        body_node = analysis.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, ERROR})
        self.assertEdge(analysis, body_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, while_node)

        else_node = analysis.edge(while_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, ERROR})
        self.assertEdge(analysis, else_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.leave_node)

    def test_while_with_two_statements(self):
        code = """\
def f():
    while some_condition:
        do_something()
        do_something_else()
"""
        analysis, while_node = self._function_analysis(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, while_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, while_node, ELSE, analysis.leave_node)

        body_node1 = analysis.edge(while_node, ENTER)
        self.assertNodetype(body_node1, ast.Expr)
        self.assertEdges(analysis, body_node1, {NEXT, ERROR})
        self.assertEdge(analysis, body_node1, ERROR, analysis.raise_node)

        body_node2 = analysis.edge(body_node1, NEXT)
        self.assertNodetype(body_node2, ast.Expr)
        self.assertEdges(analysis, body_node2, {NEXT, ERROR})
        self.assertEdge(analysis, body_node2, ERROR, analysis.raise_node)
        self.assertEdge(analysis, body_node2, NEXT, while_node)

    def test_while_true(self):
        code = """\
def f():
    while True:
        pass
"""
        analysis, while_node = self._function_analysis(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ENTER})

        pass_node = analysis.edge(while_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, while_node)

        with self.assertRaises(AttributeError):
            analysis.leave_node
        with self.assertRaises(AttributeError):
            analysis.return_node
        with self.assertRaises(AttributeError):
            analysis.raise_node

    def test_while_false(self):
        code = """\
def f():
    while False:
        pass
"""
        analysis, while_node = self._function_analysis(code)
        self.assertNodetype(while_node, ast.While)
        self.assertEdges(analysis, while_node, {ELSE})
        self.assertEdge(analysis, while_node, ELSE, analysis.leave_node)

    def test_if_true(self):
        code = """\
if True:
    pass
else:
    do_something_else()
"""
        analysis, if_node = self._module_analysis(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ENTER})

        pass_node = analysis.edge(if_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})

        self.assertEdge(analysis, pass_node, NEXT, analysis.leave_node)

    def test_if_false(self):
        code = """\
if False:
    do_something()
else:
    pass
"""
        analysis, if_node = self._module_analysis(code)
        self.assertNodetype(if_node, ast.If)
        self.assertEdges(analysis, if_node, {ELSE})

        pass_node = analysis.edge(if_node, ELSE)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})

        self.assertEdge(analysis, pass_node, NEXT, analysis.leave_node)

    def test_if_general_constant(self):
        # Check that various constants are recognised as such.
        true_constants = [
            "True",
            "2j",
            "3.0",
            "45",
            '"a string"',
            'b"some bytes"',
            "...",
        ]
        false_constants = ["False", "None", "0j", "0.0", "0", '""', 'b""']

        for true_constant in true_constants:
            code = f"""\
if {true_constant}:
    do_something()
else:
    do_something_else()
"""
            with self.subTest(code=code):
                analysis, if_node = self._module_analysis(code)
                self.assertEdges(analysis, if_node, {ENTER})

        for false_constant in false_constants:
            code = f"""\
if {false_constant}:
    do_something()
else:
    do_something_else()
"""
            with self.subTest(code=code):
                analysis, if_node = self._module_analysis(code)
                self.assertEdges(analysis, if_node, {ELSE})

    def test_assert_true(self):
        # Note that some_expression is not evaluated if the constant is true,
        # so there's no path that can raise in this case.
        code = """\
assert True, some_expression()
"""
        analysis, assert_node = self._module_analysis(code)
        self.assertNodetype(assert_node, ast.Assert)
        self.assertEdges(analysis, assert_node, {NEXT})
        self.assertEdge(analysis, assert_node, NEXT, analysis.leave_node)

    def test_assert_false(self):
        # Here the expression is evaluated, so we could end up either with
        # an AssertionError from the assert False, or with an exception
        # raised by some_expression(). Either way, we raise.
        code = """\
assert False, some_expression()
"""
        analysis, assert_node = self._module_analysis(code)
        self.assertNodetype(assert_node, ast.Assert)
        self.assertEdges(analysis, assert_node, {ERROR})
        self.assertEdge(analysis, assert_node, ERROR, analysis.raise_node)

    def test_assert_true_without_message(self):
        # Note that some_expression is not evaluated if the constant is true,
        # so there's no path that can raise in this case.
        code = """\
assert True
"""
        analysis, assert_node = self._module_analysis(code)
        self.assertNodetype(assert_node, ast.Assert)
        self.assertEdges(analysis, assert_node, {NEXT})
        self.assertEdge(analysis, assert_node, NEXT, analysis.leave_node)

    def test_assert_false_without_message(self):
        # Here the expression is evaluated, so we could end up either with
        # an AssertionError from the assert False, or with an exception
        # raised by some_expression(). Either way, we raise.
        code = """\
assert False
"""
        analysis, assert_node = self._module_analysis(code)
        self.assertNodetype(assert_node, ast.Assert)
        self.assertEdges(analysis, assert_node, {ERROR})
        self.assertEdge(analysis, assert_node, ERROR, analysis.raise_node)

    def test_assert_general(self):
        code = """\
assert some_test(), some_expression()
"""
        analysis, assert_node = self._module_analysis(code)
        self.assertNodetype(assert_node, ast.Assert)
        self.assertEdges(analysis, assert_node, {ERROR, NEXT})
        self.assertEdge(analysis, assert_node, NEXT, analysis.leave_node)
        self.assertEdge(analysis, assert_node, ERROR, analysis.raise_node)

    def test_if_else_analysed_even_if_unreachable(self):
        code = """
if True:
    do_something()
else:
    do_something_else()
"""
        self.assertAllStatementsCovered(code)

    def test_if_analysed_even_if_unreachable(self):
        code = """
if False:
    do_something()
else:
    do_something_else()
"""
        self.assertAllStatementsCovered(code)

    def test_statements_after_return_analysed(self):
        code = """
def f():
    return
    a = 23
    return True
    b = 56
    raise RuntimeError()
    c = 78
    assert False, "never get here"
    d = 123
"""
        self.assertAllFunctionStatementsCovered(code)

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
        analysis, for_node = self._function_analysis(code)
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        test_node = analysis.edge(for_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(analysis, test_node, ERROR, analysis.raise_node)
        self.assertEdges(analysis, test_node, {ENTER, ELSE, ERROR})

        continue_node = analysis.edge(test_node, ENTER)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(analysis, continue_node, {NEXT})
        self.assertEdge(analysis, continue_node, NEXT, for_node)

        body_node = analysis.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, ERROR})
        self.assertEdge(analysis, body_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, for_node)

        else_node = analysis.edge(for_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, ERROR})
        self.assertEdge(analysis, else_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.leave_node)

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
        analysis, for_node = self._function_analysis(code)
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        test_node = analysis.edge(for_node, ENTER)
        self.assertNodetype(test_node, ast.If)
        self.assertEdge(analysis, test_node, ERROR, analysis.raise_node)
        self.assertEdges(analysis, test_node, {ENTER, ELSE, ERROR})

        break_node = analysis.edge(test_node, ENTER)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(analysis, break_node, {NEXT})
        self.assertEdge(analysis, break_node, NEXT, analysis.leave_node)

        body_node = analysis.edge(test_node, ELSE)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, ERROR})
        self.assertEdge(analysis, body_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, for_node)

        else_node = analysis.edge(for_node, ELSE)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, ERROR})
        self.assertEdge(analysis, else_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.leave_node)

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
        analysis, start_node = self._function_analysis(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Expr)
        self.assertEdges(analysis, try_node, {NEXT, ERROR})

        except1_node = analysis.edge(try_node, ERROR)
        self.assertNodetype(except1_node, ast.expr)
        self.assertEdges(analysis, except1_node, {ENTER, ELSE, ERROR})
        self.assertEdge(analysis, except1_node, ERROR, analysis.raise_node)

        match1_node = analysis.edge(except1_node, ENTER)
        self.assertNodetype(match1_node, ast.Expr)
        self.assertEdges(analysis, match1_node, {NEXT, ERROR})
        self.assertEdge(analysis, match1_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, match1_node, NEXT, analysis.leave_node)

        match2_node = analysis.edge(except1_node, ELSE)
        self.assertNodetype(match2_node, ast.Expr)
        self.assertEdges(analysis, match2_node, {NEXT, ERROR})
        self.assertEdge(analysis, match2_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, match2_node, NEXT, analysis.leave_node)

        else_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(else_node, ast.Expr)
        self.assertEdges(analysis, else_node, {NEXT, ERROR})
        self.assertEdge(analysis, else_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, else_node, NEXT, analysis.leave_node)

    def test_try_except_pass(self):
        code = """\
def f():
    try:
        something_dangerous()
    except:
        pass
"""
        analysis, start_node = self._function_analysis(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Expr)
        self.assertEdges(analysis, try_node, {NEXT, ERROR})
        self.assertEdge(analysis, try_node, NEXT, analysis.leave_node)

        pass_node = analysis.edge(try_node, ERROR)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.leave_node)

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
        analysis, start_node = self._function_analysis(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Raise)
        self.assertEdges(analysis, try_node, {ERROR})

        except_node = analysis.edge(try_node, ERROR)
        self.assertNodetype(except_node, ast.expr)
        self.assertEdges(analysis, except_node, {ENTER, ELSE, ERROR})
        self.assertEdge(analysis, except_node, ELSE, analysis.raise_node)
        self.assertEdge(analysis, except_node, ERROR, analysis.raise_node)

        pass_node = analysis.edge(except_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.leave_node)

    def test_try_finally_pass(self):
        code = """\
def f():
    try:
        pass
    finally:
        do_something()
"""
        analysis, start_node = self._function_analysis(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Pass)
        self.assertEdges(analysis, try_node, {NEXT})

        finally_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.leave_node)

    def test_try_finally_raise(self):
        code = """\
def f():
    try:
        raise ValueError()
    finally:
        do_something()
"""
        analysis, start_node = self._function_analysis(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Raise)
        self.assertEdges(analysis, try_node, {ERROR})

        finally_node = analysis.edge(try_node, ERROR)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.raise_node)

    def test_try_finally_return(self):
        code = """\
def f():
    try:
        return
    finally:
        do_something()
"""
        analysis, start_node = self._function_analysis(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Return)
        self.assertEdges(analysis, try_node, {NEXT})

        finally_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.leave_node)

    def test_try_finally_return_value(self):
        code = """\
def f():
    try:
        return "abc"
    finally:
        do_something()
"""
        analysis, start_node = self._function_analysis(code)
        self.assertNodetype(start_node, ast.Try)
        self.assertEdges(analysis, start_node, {NEXT})

        try_node = analysis.edge(start_node, NEXT)
        self.assertNodetype(try_node, ast.Return)
        self.assertEdges(analysis, try_node, {NEXT, ERROR})

        finally_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.return_node)

        finally2_node = analysis.edge(try_node, ERROR)
        self.assertNodetype(finally2_node, ast.Expr)
        self.assertEdges(analysis, finally2_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally2_node, ERROR, analysis.raise_node)
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
        analysis, for_node = self._function_analysis(code)
        self.assertNodetype(for_node, ast.For)
        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertNodetype(try_node, ast.Try)
        self.assertEdges(analysis, try_node, {NEXT})

        break_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(break_node, ast.Break)
        self.assertEdges(analysis, break_node, {NEXT})

        finally_node = analysis.edge(break_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.leave_node)

    def test_try_finally_continue(self):
        code = """\
def f():
    for item in items:
        try:
            continue
        finally:
            do_something()
"""
        analysis, for_node = self._function_analysis(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        continue_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(continue_node, ast.Continue)
        self.assertEdges(analysis, continue_node, {NEXT})

        finally_node = analysis.edge(continue_node, NEXT)
        self.assertNodetype(finally_node, ast.Expr)
        self.assertEdges(analysis, finally_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, for_node)

    def test_return_value_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return some_value()
"""
        analysis, try_node = self._function_analysis(code)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {ERROR})

        return_node = analysis.edge(raise_node, ERROR)
        self.assertEdges(analysis, return_node, {NEXT, ERROR})
        self.assertEdge(analysis, return_node, NEXT, analysis.return_node)
        self.assertEdge(analysis, return_node, ERROR, analysis.raise_node)

    def test_return_in_finally(self):
        code = """\
def f():
    try:
        raise SomeException()
    finally:
        return
"""
        analysis, try_node = self._function_analysis(code)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {ERROR})

        return_node = analysis.edge(raise_node, ERROR)
        self.assertEdges(analysis, return_node, {NEXT})
        self.assertEdge(analysis, return_node, NEXT, analysis.leave_node)

    def test_raise_in_finally(self):
        code = """\
def f():
    try:
        pass
    finally:
        raise SomeException()
"""
        analysis, try_node = self._function_analysis(code)
        self.assertEdges(analysis, try_node, {NEXT})

        pass_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})

        raise_node = analysis.edge(pass_node, NEXT)
        self.assertNodetype(raise_node, ast.Raise)
        self.assertEdges(analysis, raise_node, {ERROR})
        self.assertEdge(analysis, raise_node, ERROR, analysis.raise_node)

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
        analysis, for_node = self._function_analysis(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        return_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, return_node, {NEXT})

        break_node = analysis.edge(return_node, NEXT)
        self.assertEdges(analysis, break_node, {NEXT})
        self.assertEdge(analysis, break_node, NEXT, analysis.leave_node)

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
        analysis, for_node = self._function_analysis(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {ERROR})

        continue_node = analysis.edge(raise_node, ERROR)
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
        analysis, for_node = self._function_analysis(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {ERROR})

        continue_node = analysis.edge(raise_node, ERROR)
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
        analysis, for_node = self._function_analysis(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {ERROR})

        continue_node = analysis.edge(raise_node, ERROR)
        self.assertEdges(analysis, continue_node, {NEXT})

        finally_node = analysis.edge(continue_node, NEXT)
        self.assertEdges(analysis, finally_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, for_node)

    def test_break_in_inner_loop(self):
        code = """\
def f():
    while some_condition():
        while some_other_condition():
            break
"""
        analysis, while_node = self._function_analysis(code)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, while_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, while_node, ERROR, analysis.raise_node)

        inner_while_node = analysis.edge(while_node, ENTER)
        self.assertEdges(analysis, inner_while_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, inner_while_node, ELSE, while_node)
        self.assertEdge(analysis, inner_while_node, ERROR, analysis.raise_node)

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
        analysis, while_node = self._function_analysis(code)
        self.assertEdges(analysis, while_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, while_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, while_node, ERROR, analysis.raise_node)

        inner_while_node = analysis.edge(while_node, ENTER)
        self.assertEdges(analysis, inner_while_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, inner_while_node, ELSE, while_node)
        self.assertEdge(analysis, inner_while_node, ERROR, analysis.raise_node)

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
        analysis, for_node = self._function_analysis(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        raise_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, raise_node, {ERROR})

        except_node = analysis.edge(raise_node, ERROR)
        self.assertEdges(analysis, except_node, {ENTER, ELSE, ERROR})
        self.assertEdge(
            analysis,
            except_node,
            ERROR,
            analysis.edge(except_node, ELSE),
        )

        finally_raise_node = analysis.edge(except_node, ERROR)
        self.assertEdges(analysis, finally_raise_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_raise_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_raise_node, NEXT, analysis.raise_node)

        break_node = analysis.edge(except_node, ENTER)
        self.assertEdges(analysis, break_node, {NEXT})

        finally_node = analysis.edge(break_node, NEXT)
        self.assertEdges(analysis, finally_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.leave_node)

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
        analysis, for_node = self._function_analysis(code)

        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        try_node = analysis.edge(for_node, ENTER)
        self.assertEdges(analysis, try_node, {NEXT})

        do_node = analysis.edge(try_node, NEXT)
        self.assertEdges(analysis, do_node, {NEXT, ERROR})

        pass_node = analysis.edge(do_node, ERROR)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})

        finally1_node = analysis.edge(pass_node, NEXT)
        self.assertEdges(analysis, finally1_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally1_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally1_node, NEXT, for_node)

        else_node = analysis.edge(do_node, NEXT)
        self.assertEdges(analysis, else_node, {NEXT})

        finally_node = analysis.edge(else_node, NEXT)
        self.assertEdges(analysis, finally_node, {NEXT, ERROR})
        self.assertEdge(analysis, finally_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, finally_node, NEXT, analysis.leave_node)

    def test_finally_analysed_even_if_not_reachable(self):
        code = """\
try:
    while True:
        pass
finally:
    assert False, "never get here"
"""
        analysis, _ = self._module_analysis(code)
        assert_nodes = [
            node
            for node in analysis.nodes()
            if hasattr(node, "ast_node")
            if isinstance(node.ast_node, ast.Assert)
        ]
        self.assertEqual(len(assert_nodes), 1)

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
        analysis, try_node = self._function_analysis(code)

        # The 'return' in the else branch should lead to the same place
        # as the handle_exception() success in the except branch.
        do_node = analysis.edge(try_node, NEXT)
        raised_node = analysis.edge(do_node, ERROR)
        ok_node = analysis.edge(do_node, NEXT)

        raised_next = analysis.edge(raised_node, NEXT)
        ok_next = analysis.edge(ok_node, NEXT)

        self.assertEqual(raised_next, ok_next)

    def test_empty_module(self):
        code = ""
        analysis, enter_node = self._module_analysis(code)
        self.assertEqual(enter_node, analysis.leave_node)

    def test_just_pass(self):
        code = """\
pass
"""
        analysis, enter_node = self._module_analysis(code)
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
        analysis, assign_node = self._module_analysis(code)
        self.assertNodetype(assign_node, ast.Assign)
        self.assertEdges(analysis, assign_node, {NEXT, ERROR})
        self.assertEdge(analysis, assign_node, ERROR, analysis.raise_node)

        try_node = analysis.edge(assign_node, NEXT)
        self.assertNodetype(try_node, ast.Try)
        self.assertEdges(analysis, try_node, {NEXT})

        do_node = analysis.edge(try_node, NEXT)
        self.assertNodetype(do_node, ast.Expr)
        self.assertEdges(analysis, do_node, {NEXT, ERROR})
        self.assertEdge(analysis, do_node, NEXT, analysis.leave_node)

        pass_node = analysis.edge(do_node, ERROR)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.leave_node)

    def test_with(self):
        code = """\
with some_cm() as name:
    do_something()
"""
        analysis, with_node = self._module_analysis(code)
        self.assertNodetype(with_node, ast.With)
        self.assertEdges(analysis, with_node, {ENTER, ERROR})
        self.assertEdge(analysis, with_node, ERROR, analysis.raise_node)

        body_node = analysis.edge(with_node, ENTER)
        self.assertNodetype(body_node, ast.Expr)
        self.assertEdges(analysis, body_node, {NEXT, ERROR})
        self.assertEdge(analysis, body_node, ERROR, analysis.raise_node)
        self.assertEdge(analysis, body_node, NEXT, analysis.leave_node)

    def test_async_for(self):
        code = """\
async def f():
    async for x in g():
        yield x*x
"""
        analysis, for_node = self._function_analysis(code)
        self.assertNodetype(for_node, ast.AsyncFor)
        self.assertEdges(analysis, for_node, {ELSE, ENTER, ERROR})
        self.assertEdge(analysis, for_node, ELSE, analysis.leave_node)
        self.assertEdge(analysis, for_node, ERROR, analysis.raise_node)

        yield_node = analysis.edge(for_node, ENTER)
        self.assertNodetype(yield_node, ast.Expr)
        self.assertEdges(analysis, yield_node, {NEXT, ERROR})
        self.assertEdge(analysis, yield_node, NEXT, for_node)
        self.assertEdge(analysis, yield_node, ERROR, analysis.raise_node)

    def test_async_with(self):
        code = """\
async def f():
    async with my_async_context():
        pass
"""
        analysis, with_node = self._function_analysis(code)
        self.assertNodetype(with_node, ast.AsyncWith)
        self.assertEdges(analysis, with_node, {ENTER, ERROR})
        self.assertEdge(analysis, with_node, ERROR, analysis.raise_node)

        pass_node = analysis.edge(with_node, ENTER)
        self.assertNodetype(pass_node, ast.Pass)
        self.assertEdges(analysis, pass_node, {NEXT})
        self.assertEdge(analysis, pass_node, NEXT, analysis.leave_node)

    def test_classdef(self):
        code = """\
class SomeClass:
    def some_method(self, arg1, arg2):
        return bob
"""
        analysis, initial = self._class_analysis(code)
        self.assertNodetype(initial, ast.FunctionDef)
        self.assertEdges(analysis, initial, {NEXT, ERROR})
        self.assertEdge(analysis, initial, NEXT, analysis.leave_node)
        self.assertEdge(analysis, initial, ERROR, analysis.raise_node)

    def test_global(self):
        code = """\
def f():
    global bob
"""
        analysis, _ = self._function_analysis(code)
        node = analysis.entry_node
        self.assertNodetype(node, ast.Global)
        self.assertEdges(analysis, node, {NEXT})
        self.assertEdge(analysis, node, NEXT, analysis.leave_node)

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
        self.assertEdge(analysis, node, NEXT, analysis.leave_node)

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
        analysis, node = self._module_analysis(code)

        for _ in range(9):
            self.assertNodetype(node, ast.stmt)
            self.assertEdges(analysis, node, {NEXT, ERROR})
            self.assertEdge(analysis, node, ERROR, analysis.raise_node)
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
        analysis, _ = self._function_analysis(code)
        with self.assertRaises(AttributeError):
            analysis.raise_node

    def test_function_no_return_value(self):
        code = """\
def f():
    pass
"""
        analysis, _ = self._function_analysis(code)
        with self.assertRaises(AttributeError):
            analysis.return_node

    def test_function_no_leave_node(self):
        code = """\
def f():
    return 123
"""
        analysis, _ = self._function_analysis(code)
        with self.assertRaises(AttributeError):
            analysis.leave_node

    def test_class_cant_raise(self):
        code = """\
class A:
    try:
        something_or_other()
    except:
        pass
"""
        analysis, _ = self._class_analysis(code)
        with self.assertRaises(AttributeError):
            analysis.raise_node

    def test_module_cant_raise(self):
        code = """\
try:
    something_or_other()
except:
    pass
"""
        analysis, _ = self._module_analysis(code)
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

    def assertAllStatementsCovered(self, code):
        """
        Check that all statements in the given code are covered
        by the analysis.
        """
        tree = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        self.assertIsInstance(tree, ast.Module)
        analysis = CFAnalyser().analyse_module(tree)
        self.assertEqual(analysed_statements(analysis), all_statements(tree))

    def assertAllFunctionStatementsCovered(self, code):
        """
        Check that all statements in the given code are covered
        by the analysis.
        """
        tree = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST).body[0]
        self.assertIsInstance(tree, ast.FunctionDef)
        analysis = CFAnalyser().analyse_function(tree)
        self.assertEqual(analysed_statements(analysis), all_statements(tree) - {tree})

    # Helper methods

    def _function_analysis(self, code):
        function_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST).body[0]
        self.assertIsInstance(function_node, (ast.AsyncFunctionDef, ast.FunctionDef))

        analysis = CFAnalyser().analyse_function(function_node)
        if hasattr(analysis, "raise_node"):
            self.assertEdges(analysis, analysis.raise_node, set())
        if hasattr(analysis, "leave_node"):
            self.assertEdges(analysis, analysis.leave_node, set())
        if hasattr(analysis, "return_node"):
            self.assertEdges(analysis, analysis.return_node, set())

        return analysis, analysis.entry_node

    def _module_analysis(self, code):
        module_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST)
        self.assertIsInstance(module_node, ast.Module)

        analysis = CFAnalyser().analyse_module(module_node)

        if hasattr(analysis, "raise_node"):
            self.assertEdges(analysis, analysis.raise_node, set())
        if hasattr(analysis, "leave_node"):
            self.assertEdges(analysis, analysis.leave_node, set())

        return analysis, analysis.entry_node

    def _class_analysis(self, code):
        class_node = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST).body[0]
        self.assertIsInstance(class_node, ast.ClassDef)

        analysis = CFAnalyser().analyse_class(class_node)

        if hasattr(analysis, "raise_node"):
            self.assertEdges(analysis, analysis.raise_node, set())
        if hasattr(analysis, "leave_node"):
            self.assertEdges(analysis, analysis.leave_node, set())

        return analysis, analysis.entry_node
