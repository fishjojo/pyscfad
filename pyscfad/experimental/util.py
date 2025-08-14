# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import ast

class RewriteName(ast.NodeTransformer):
    def __init__(self, orig, repl):
        self.orig = orig
        self.repl = repl

    def visit_Name(self, node):
        if node.id == self.orig:
            result = ast.Name(id=self.repl, ctx=ast.Load())
            return ast.copy_location(result, node)
        return node

def replace_source_code(fn, namespace, orig, repl):
    # FIXME the returned function when being inspected gives wrong result
    source = inspect.getsource(fn)
    tree = ast.parse(source)
    new_tree = ast.fix_missing_locations(RewriteName(orig, repl).visit(tree))
    code = compile(new_tree, inspect.getfile(fn), 'exec')
    # pylint: disable = exec-used
    exec(code, namespace)
    return namespace[fn.__name__]

def numpy2np(fn, namespace=None, np='np'):
    if namespace is None:
        namespace = fn.__globals__
    return replace_source_code(fn, namespace, 'numpy', np)
