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

import ast

import pygraphviz as pgv

from pycfa.cf import CFAnalyser

code = """\
def f():
    a = 1
    b = 2
    for _ in range(10):
        if some_condition:
            continue
        elif other_condition:
            break
    else:
        pass
"""
lines = code.splitlines(keepends=True)


(function_node,) = compile(code, "test_cf", "exec", ast.PyCF_ONLY_AST).body
analysis = CFAnalyser.from_function(function_node)

graph = analysis._graph

G = pgv.AGraph(strict=False, directed=True)

for node in graph._nodes:
    kwds = dict(shape="box")
    if node.ast_node is not None:
        lineno = node.ast_node.lineno
        kwds.update(label=f"{lineno}: {lines[node.ast_node.lineno - 1].strip()}")
    elif node.annotation is not None:
        kwds.update(label=node.annotation)
    G.add_node(id(node), **kwds)

for source, out_edges in graph._edges.items():
    for label, target in out_edges.items():
        # Figure out whether this edge goes backwards w.r.t. line number
        kwds = {}
        if source.ast_node is not None and target.ast_node is not None:
            if source.ast_node.lineno > target.ast_node.lineno:
                kwds["constraint"] = "false"
        G.add_edge(id(source), id(target), label=label, **kwds)


print(G.string())
G.layout(prog="dot")
G.draw("output.svg")
