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
Tests for the flake8 extension to find redundant returns.
"""

import importlib.resources
import subprocess
import unittest

# XXX Tests assume that the extension is installed in the local environment.
#     Can we check for that and skip if not true?
# XXX Make sure everything is typed!
# XXX Need to update dependencies in setup file.
# XXX Use importlib_resources for now, while we still need Python 3.6 compatibility.
# XXX Make it possible to re-use CFAnalyser (or at least raise if we try to use
#     it twice in a row, instead of augmenting the graph.)
# XXX Make checker version match the package version.
# XXX Test plugin directly as well as through flake8.
# XXX Fix analysis of finally branches: a return at the end of a finally branch
#     is redundant only if it's redundant for _all_ paths through that finally
#     branch.
# XXX Don't raise on 'return constant'.


# Mypy currently complains about the non-existence of "files".
TEST_RESOURCES = importlib.resources.files("pycfa.test")  # type: ignore[attr-defined]
TEST_SCRIPTS = TEST_RESOURCES / "scripts"


class TestRedundantReturns(unittest.TestCase):
    def test_redundant_return_simple_cases(self):
        filename = "martin_return.py"
        cmd = ["flake8", "--select=MCR", filename]
        result = subprocess.run(cmd, capture_output=True, cwd=TEST_SCRIPTS)
        reported_errors = result.stdout.decode("utf-8").splitlines()
        expected_errors = [
            f"{filename}:21:5: MCR100 Redundant return",
            f"{filename}:27:9: MCR100 Redundant return",
            f"{filename}:49:9: MCR100 Redundant return",
        ]
        self.assertEqual(reported_errors, expected_errors)

    def test_redundant_return_in_if_and_else(self):
        filename = "return_in_if.py"
        cmd = ["flake8", filename]
        result = subprocess.run(cmd, capture_output=True, cwd=TEST_SCRIPTS)
        reported_errors = result.stdout.decode("utf-8").splitlines()
        expected_errors = [
            f"{filename}:22:9: MCR100 Redundant return",
            f"{filename}:25:9: MCR100 Redundant return",
            f"{filename}:31:9: MCR100 Redundant return",
            f"{filename}:39:9: MCR100 Redundant return",
        ]
        self.assertEqual(reported_errors, expected_errors)

    def test_redundant_return_in_try_except(self):
        filename = "return_in_try_except.py"
        cmd = ["flake8", filename]
        result = subprocess.run(cmd, capture_output=True, cwd=TEST_SCRIPTS)
        reported_errors = result.stdout.decode("utf-8").splitlines()
        expected_errors = [
            f"{filename}:24:9: MCR100 Redundant return",
            f"{filename}:26:9: MCR100 Redundant return",
            f"{filename}:28:9: MCR100 Redundant return",
        ]
        self.assertEqual(reported_errors, expected_errors)
