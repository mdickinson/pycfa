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


import random


def has_redundant_return_both_branches():
    if random.random() < 0.5:
        print("If branch")
        return
    else:
        print("Else branch")
        return


def has_redundant_return_if_only():
    if random.random() < 0.5:
        print("If branch")
        return
    else:
        print("Else branch")


def has_redundant_return_else_only():
    if random.random() < 0.5:
        print("If branch")
        return
    else:
        print("Else branch")