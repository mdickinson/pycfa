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


def has_redundant_return():
    print("Hello world")
    return


class A:
    def __init__(self):
        self.some_attr = 3
        return


def not_redundant_return(self):
    if random.random() < 0.5:
        print("Early return")
        return
    print("Normal execution")


def not_redundant_return_with_value(self):
    print("Do something")
    return 3


def redundant_return_in_for_else(self):
    for i in range(3):
        if random.random() < 0.5:
            print("Found: ", i)
            return
    else:
        print("Not found")
        return
