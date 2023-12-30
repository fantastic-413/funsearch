# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for evaluating programs proposed by the Sampler."""
import ast
from collections.abc import Sequence
import copy
from typing import Any

from funsearch.implementation import code_manipulation
from funsearch.implementation import programs_database

# _FunctionLineVisitor类的作用是遍历一个 ast.Node 对象，找到目标函数的最后一行的行号
# _FunctionLineVisitor类继承自 ast.NodeVisitor 类
class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  # __init__函数接收一个字符串，表示要查找的函数的名字
  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    # _function_end_line属性的作用是存储要查找的函数的最后一行的行号
    # 这个属性的值会在 visit_FunctionDef 函数中被赋值
    self._function_end_line: int | None = None

  # visit_FunctionDef 函数接收一个 ast.FunctionDef 对象
  # ast.FunctionDef 对象表示一个函数定义
  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    # 如果 node.name == self._target_function_name，说明找到了目标函数
    # 这时会将目标函数的最后一行的行号赋值给 self._function_end_line
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    # generic_visit 函数的作用是遍历 node 的所有子节点
    self.generic_visit(node)

  # function_end_line属性的作用是存储目标函数的最后一行的行号
  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    # 这行代码的作用是检查 self._function_end_line 是否为 None
    # 如果为 None，说明没有找到目标函数，会抛出一个 ValueError 异常
    # 如果不为 None，说明找到了目标函数，会返回 self._function_end_line
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line

# _trim_function_body 函数的作用是从 generated_code 中提取出函数体，并返回一个字符串
def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  # 将 generated_code 的内容放到一个字符串中，然后在字符串的开头添加一行代码，这行代码定义了一个函数 fake_function_header
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  # 这个循环的作用是将 code 解析成一个 ast.Node 对象
  # 如果解释失败，会通过e.lineno来获取错误的行号，然后通过code.splitlines()[:e.lineno - 1]来获取错误行之前的所有代码
  # 然后再次尝试解析，直到解析成功
  while tree is None:
    # 将 code 解析成一个 ast.Node 对象
    try:
      tree = ast.parse(code)
    # 如果解析失败，就会抛出一个 SyntaxError 异常
    # 这个异常包含了错误的行号，可以通过 e.lineno 来获取
    # 通过 e.lineno 可以知道哪一行的代码导致了解析失败
    # 然后通过 code.splitlines()[:e.lineno - 1] 来获取错误行之前的所有代码
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  # visitor被定义为一个 _FunctionLineVisitor 类的实例，用于遍历 tree
  visitor = _FunctionLineVisitor('fake_function_header')
  # visitor.visit(tree)的作用是遍历 tree，找到目标函数的最后一行的行号
  visitor.visit(tree)
  # body_lines的作用是存储目标函数的所有代码
  body_lines = code.splitlines()[1:visitor.function_end_line]
  # 这行代码的作用是将 body_lines 中的所有代码连接成一个字符串，并返回
  return '\n'.join(body_lines) + '\n\n'


# _sample_to_program 函数接收 4 个参数
# 1. generated_code：一个字符串，包含了一个continuation
# 2. version_generated：一个整数，表示continuation的版本号
# 3. template：一个 Program 对象，包含了要evolve的函数和要run的函数
# 4. function_to_evolve：一个字符串，表示要evolve的函数的名字
# 返回类型被定义为 tuple[Function, str]，意味着这个函数返回一个包含两个字符串的元组。
# 第一个字符串是一个 Function 对象，包含了要evolve的函数的所有信息
# 第二个字符串是一个 Program 对象，包含了要evolve的函数和要run的函数
def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program."""
  # _trim_function_body 函数的作用是从 generated_code 中提取出函数体，并返回一个字符串
  body = _trim_function_body(generated_code)
  # 如果 version_generated 不为 None，说明 generated_code 是一个continuation
  # 这是会通过 code_manipulation.rename_function_calls 函数来修改 body 中的函数调用
  # 将函数调用从 function_to_evolve_v{version_generated} 修改为 function_to_evolve
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  # copy.deepcopy(template)的作用是创建一个 template 的深拷贝
  # 这个深拷贝包含了要evolve的函数和要run的函数
  # 这个深拷贝的作用是防止修改 template
  program = copy.deepcopy(template)
  # program.get_function(function_to_evolve)的作用是获取要evolve的函数的 Function 对象
  evolved_function = program.get_function(function_to_evolve)
  # 通过修改 evolved_function.body 来修改要evolve的函数的函数体
  evolved_function.body = body
  return evolved_function, str(program)

# Sandbox类的作用是执行生成的代码
class Sandbox:
  """Sandbox for executing generated code."""

  # run函数接收 5 个参数，然后返回一个元组
  # 1. program：一个字符串，包含了要evolve的函数和要run的函数
  # 2. function_to_run：一个字符串，表示要run的函数的名字
  # 3. test_input：一个字符串，表示要run的函数的参数
  # 4. timeout_seconds：一个整数，表示执行代码的超时时间
  # 返回类型被定义为 tuple[Any, bool]，意味着这个函数返回一个包含两个元素的元组。
  # 第一个元素是要run的函数的返回值
  # 第二个元素是一个布尔值，表示执行代码是否成功
  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: str,
      timeout_seconds: int,
  ) -> tuple[Any, bool]:
    # 返回`function_to_run(test_input)`和执行是否成功。
    """Returns `function_to_run(test_input)` and whether execution succeeded."""
    # 必须提供一个沙箱来执行不受信任的代码。
    raise NotImplementedError(
        'Must provide a sandbox for executing untrusted code.')

# _calls_ancestor 函数的作用是返回生成的函数是否调用了一个早期版本的函数
def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  # 这个循环的作用是遍历 program 中的所有函数调用
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    # 在传递给这个函数的 `program` 中，最近生成的函数已经被重命名为 `function_to_evolve`（没有后缀）。
    # 因此，任何以 `function_to_evolve_v` 开头的函数调用都是对祖先函数的调用。
    # 如果 name 以 function_to_evolve_v 开头，说明生成的函数调用了一个早期版本的函数
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


# Evaluator类的作用是分析一个函数实现的性能
# Evaluator类的构造函数接受 5 个参数
# 1. ProgramsDatabase 对象，用于管理所有的 Function 对象
# 2. Program 对象，包含了要evolve的函数和要run的函数
# 3. 要evolve的函数名，用于从 Program 对象中获取要evolve的函数
# 4. 要run的函数名，用于从 Program 对象中获取要run的函数
# 5. 要run的函数的参数，用于在 Sandbox 中执行要run的函数
class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],
      timeout_seconds: int = 30,
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = Sandbox()

  # analyse函数接收sample, prompt.island_id, prompt.version_generated三个参数
  # sample是一个字符串,包含了一个continuation
  # prompt.island_id是一个整数,表示prompt所在的岛屿的id
  # prompt.version_generated是一个整数,表示prompt的版本号
  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
  ) -> None:
    """Compiles the sample into a program and executes it on test inputs."""
    # _sample_to_program 函数的作用是将 sample 转换成一个 Function 对象和一个 Program 对象
    # new_function是一个 Function 对象，包含了要evolve的函数的所有信息
    # program是一个 Program 对象，包含了要evolve的函数和要run的函数
    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)

    # We only register the program if it passes the sandbox.
    # 我们只有在程序通过沙箱时才会注册程序。
    scores_per_test = {}
    for current_input in self._inputs:
      # 这行代码的作用是在 Sandbox 中执行要run的函数
      # test_output是要run的函数的返回值
      # runs_ok是一个布尔值，表示执行代码是否成功
      test_output, runs_ok = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds)
      # 如果 runs_ok 为 True 且 new_function 没有调用早期版本的函数 且 test_output 不为 None
      # 将 test_output 添加到 scores_per_test 中
      if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
          and test_output is not None):
        # 如果 test_output 不是一个 int 或者 float，会抛出一个 ValueError 异常
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        # 将 test_output 添加到 scores_per_test 中
        scores_per_test[current_input] = test_output
    # 如果 scores_per_test 不为空，说明生成的函数通过了沙箱，可以注册到数据库中
    if scores_per_test:
      self._database.register_program(new_function, island_id, scores_per_test)
