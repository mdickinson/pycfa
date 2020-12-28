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
Tests for the CFGraph structure.
"""


import unittest

from pycfa.cfgraph import CFGraph


class TestCFGraph(unittest.TestCase):
    def test_add_node_with_edges(self):
        graph: CFGraph[int] = CFGraph()
        graph.add_node(23, edges={})
        self.assertIn(23, graph)

        graph.add_node(47, edges={"next": 23})
        self.assertIn(47, graph)

    def test_add_node_edges_parameter_is_optional(self):
        graph: CFGraph[int] = CFGraph()
        graph.add_node(23)
        self.assertIn(23, graph)

    def test_add_node_edges_parameter_is_keyword_only(self):
        # The edges parameter must be passed by name.
        graph: CFGraph[int] = CFGraph()
        with self.assertRaises(TypeError):
            graph.add_node(23, {})
        self.assertNotIn(23, graph)

    def test_add_node_twice(self):
        graph: CFGraph[int] = CFGraph()
        graph.add_node(23)
        with self.assertRaises(ValueError):
            graph.add_node(23)

    def test_add_node_with_self_edge(self):
        graph: CFGraph[int] = CFGraph()
        with self.assertRaises(ValueError):
            graph.add_node(23, edges={"self": 23})

    def test_add_node_edge_to_nonexistent_node(self):
        graph: CFGraph[int] = CFGraph()
        graph.add_node(23)
        with self.assertRaises(ValueError):
            graph.add_node(47, edges={"next": 48})

    def test_remove_node(self):
        graph: CFGraph[int] = CFGraph()
        graph.add_node(47)
        self.assertIn(47, graph)
        graph.remove_node(47)
        self.assertNotIn(47, graph)

    def test_remove_nonexistent_node(self):
        graph: CFGraph[int] = CFGraph()
        with self.assertRaises(ValueError):
            graph.remove_node(47)
        self.assertNotIn(47, graph)

    def test_remove_node_with_forward_edges(self):
        graph: CFGraph[int] = CFGraph()

        graph.add_node(24)
        graph.add_node(23, edges={"next": 24})
        with self.assertRaises(ValueError):
            graph.remove_node(23)
        self.assertIn(23, graph)

    def test_remove_node_with_back_edges(self):
        graph: CFGraph[int] = CFGraph()

        graph.add_node(24)
        graph.add_node(23, edges={"next": 24})
        with self.assertRaises(ValueError):
            graph.remove_node(24)
        self.assertIn(24, graph)

    def test_collapse_node(self):
        graph: CFGraph[int] = CFGraph()

        graph.add_node(3)

        graph.add_node(2)
        graph.add_node(1, edges={"step": 2})
        graph.add_node(0, edges={"next": 2})

        self.assertIn(2, graph)
        self.assertEqual(graph.edge_labels(2), set())
        self.assertEqual(graph.edge(1, "step"), 2)
        self.assertEqual(graph.edge(0, "next"), 2)
        self.assertEqual(graph.edges_to(2), {(0, "next"), (1, "step")})
        self.assertEqual(graph.edges_to(3), set())

        graph.collapse_node(2, 3)

        self.assertNotIn(2, graph)
        self.assertEqual(graph.edge_labels(1), {"step"})
        self.assertEqual(graph.edge(1, "step"), 3)
        self.assertEqual(graph.edge(0, "next"), 3)
        self.assertEqual(graph.edges_to(3), {(0, "next"), (1, "step")})

    def test_collapse_node_with_forward_edges(self):
        graph: CFGraph[int] = CFGraph()

        graph.add_node(3)

        graph.add_node(2)
        graph.add_node(1, edges={"step": 2})
        with self.assertRaises(ValueError):
            graph.collapse_node(1, 3)

    def test_collapse_node_with_bad_node(self):
        graph: CFGraph[int] = CFGraph()
        graph.add_node(3)

        with self.assertRaises(ValueError):
            graph.collapse_node(1, 3)

        with self.assertRaises(ValueError):
            graph.collapse_node(3, 1)

    def test_collapse_with_identical_nodes(self):
        graph: CFGraph[int] = CFGraph()
        graph.add_node(3)

        with self.assertRaises(ValueError):
            graph.collapse_node(3, 3)
