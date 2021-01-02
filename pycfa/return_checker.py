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

from pycfa.cfanalyser import CFAnalyser


class RedundantReturnError:
    """
    Error reported for redundant return statements.
    """


class ReturnChecker(object):
    """
    Flake8 plugin for checking for redundant returns.
    """

    #: Name of the plugin, which appears when one does 'flake8 --version'
    #: on the command line.
    name = "return_checker"

    #: Version of the extension.
    version = "0.1.0"

    def __init__(self, tree):
        self.tree = tree

    def run(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                analysis = CFAnalyser().analyse_function(node)
                for node in analysis.redundant_returns:
                    ast_node = node.ast_node
                    yield (
                        ast_node.lineno,
                        ast_node.col_offset,
                        "MCR100 Redundant return",
                        type(self),
                    )
