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

"""Tools for manipulating Python code.

It implements 2 classes representing unities of code:
- Function, containing all the information we need about functions: name, args,
  body and optionally a return type and a docstring.
- Program, which contains a code preface (which could be imports, global
  variables and classes, ...) and a list of Functions.
"""
import ast
from collections.abc import Iterator, MutableSet, Sequence
import dataclasses
import io
import tokenize

from absl import logging


@dataclasses.dataclass
class Function:
  """A parsed Python function."""

  name: str
  args: str
  body: str
  return_type: str | None = None
  docstring: str | None = None

  def __str__(self) -> str:
    return_type = f' -> {self.return_type}' if self.return_type else ''

    function = f'def {self.name}({self.args}){return_type}:\n'
    if self.docstring:
      # self.docstring is already indented on every line except the first one.
      # Here, we assume the indentation is always two spaces.
      new_line = '\n' if self.body else ''
      function += f'  """{self.docstring}"""{new_line}'
    # self.body is already indented.
    function += self.body + '\n\n'
    return function

  def __setattr__(self, name: str, value: str) -> None:
    # Ensure there aren't leading & trailing new lines in `body`.
    if name == 'body':
      value = value.strip('\n')
    # Ensure there aren't leading & trailing quotes in `docstring``.
    if name == 'docstring' and value is not None:
      if '"""' in value:
        value = value.strip()
        value = value.replace('"""', '')
    super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
  """A parsed Python program."""

  # `preface` is everything from the beginning of the code till the first
  # function is found.
  preface: str
  functions: list[Function]

  def __str__(self) -> str:
    program = f'{self.preface}\n' if self.preface else ''
    program += '\n'.join([str(f) for f in self.functions])
    return program

  def find_function_index(self, function_name: str) -> int:
    """Returns the index of input function name."""
    function_names = [f.name for f in self.functions]
    count = function_names.count(function_name)
    if count == 0:
      raise ValueError(
          f'function {function_name} does not exist in program:\n{str(self)}'
      )
    if count > 1:
      raise ValueError(
          f'function {function_name} exists more than once in program:\n'
          f'{str(self)}'
      )
    index = function_names.index(function_name)
    return index

  def get_function(self, function_name: str) -> Function:
    index = self.find_function_index(function_name)
    return self.functions[index]


class ProgramVisitor(ast.NodeVisitor):
  """Parses code to collect all required information to produce a `Program`.

  Note that we do not store function decorators.
  """

  def __init__(self, sourcecode: str):
    self._codelines: list[str] = sourcecode.splitlines()

    self._preface: str = ''
    self._functions: list[Function] = []
    self._current_function: str | None = None

  def visit_FunctionDef(self,  # pylint: disable=invalid-name
                        node: ast.FunctionDef) -> None:
    """Collects all information about the function being parsed."""
    if node.col_offset == 0:  # We only care about first level functions.
      self._current_function = node.name
      if not self._functions:
        self._preface = '\n'.join(self._codelines[:node.lineno - 1])
      function_end_line = node.end_lineno
      body_start_line = node.body[0].lineno - 1
      # Extract the docstring.
      docstring = None
      if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value,
                                                           ast.Str):
        docstring = f'  """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
        if len(node.body) > 1:
          body_start_line = node.body[1].lineno - 1
        else:
          body_start_line = function_end_line

      self._functions.append(Function(
          name=node.name,
          args=ast.unparse(node.args),
          return_type=ast.unparse(node.returns) if node.returns else None,
          docstring=docstring,
          body='\n'.join(self._codelines[body_start_line:function_end_line]),
      ))
    self.generic_visit(node)

  def return_program(self) -> Program:
    return Program(preface=self._preface, functions=self._functions)

# 通过使用 Python AST 解析输入文本，返回 Program 对象。
# 这个函数接受一个字符串 text 作为参数，这通常是一段 Python 代码
# 它返回一个 Program 类型的对象，这个 Program 类型可能是自定义的，用于表示程序的结构或内容。
def text_to_program(text: str) -> Program:
  """Returns Program object by parsing input text using Python AST."""
  try:
    # We assume that the program is composed of some preface (e.g. imports,
    # classes, assignments, ...) followed by a sequence of functions.
    # 这行代码使用 Python 的 ast 模块将文本解析为一个 AST（抽象语法树）。AST 是一种树状的数据结构，用于表示程序代码的结构。
    tree = ast.parse(text)
    # 这行代码使用 ProgramVisitor 类来遍历 AST，从而收集所有需要的信息来生成一个 Program 对象。
    visitor = ProgramVisitor(text)
    visitor.visit(tree)
    # 这行代码返回一个 Program 对象。
    return visitor.return_program()
  except Exception as e:
    logging.warning('Failed parsing %s', text)
    raise e

# 通过使用 Python AST 解析输入文本，返回 Function 对象。
def text_to_function(text: str) -> Function:
  """Returns Function object by parsing input text using Python AST."""
  # 这行代码使用 text_to_program 函数将 text 转换为 Program 对象。
  program = text_to_program(text)
  # 这行代码检查 Program 对象中的函数数量是否为 1，如果不是，就抛出一个 ValueError 异常。
  if len(program.functions) != 1:
    raise ValueError(f'Only one function expected, got {len(program.functions)}'
                     f':\n{program.functions}')
  # 这行代码返回 Program 对象中的第一个函数。
  return program.functions[0]

# 将 code 转换为 Python 令牌。
# 返回类型被定义为 Iterator[tokenize.TokenInfo]，意味着这个函数返回一个迭代器，迭代器的每个元素都是一个 tokenize.TokenInfo 对象。
def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
  """Transforms `code` into Python tokens."""
  code_bytes = code.encode() # 将 code 转换为字节
  code_io = io.BytesIO(code_bytes) # 将字节转换为 io.BytesIO 对象
  return tokenize.tokenize(code_io.readline) # 将 io.BytesIO 对象转换为 Python 令牌

# 将 Python 令牌转换为 code。
def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
  """Transforms a list of Python tokens into code."""
  code_bytes = tokenize.untokenize(tokens) # 将 Python 令牌转换为字节
  return code_bytes.decode() # 将字节转换为字符串

# 这个函数接受一个字符串 code 作为参数，这通常是一段 Python 代码
# 函数作用是通过_tokenize函数将 code 转换为 Python 令牌，然后遍历 Python 令牌，找到所有的函数调用
# 它返回一个迭代器，迭代器的每个元素都是一个元组，元组的第一个元素是一个 tokenize.TokenInfo 对象，第二个元素是一个布尔值，表示这个 TokenInfo 对象是否是一个函数调用。
def _yield_token_and_is_call(
    code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
  """Yields each token with a bool indicating whether it is a function call."""
  try:
    # 这行代码使用 _tokenize 函数将 code 转换为 Python 令牌。
    tokens = _tokenize(code)
    prev_token = None # prev_token 的作用是存储上一个 Python 令牌
    is_attribute_access = False # is_attribute_access 的作用是存储当前 Python 令牌是否是一个属性访问
    # 这个循环遍历 tokens 中的所有 Python 令牌
    for token in tokens:
      # 如果 prev_token 存在，且 prev_token 是一个 Python 标识符，且 token 是一个分隔符，且 token 是 '('。
      # 这个条件的作用是判断 token 是否是一个函数调用的左括号。
      if (prev_token and  # If the previous token exists and
          prev_token.type == tokenize.NAME and  # it is a Python identifier
          token.type == tokenize.OP and  # and the current token is a delimiter
          token.string == '('):  # and in particular it is '('.
        # 通过 yield 语句将 prev_token 和 is_attribute_access 作为一个元组返回。
        # not is_attribute_access 表示如果当前 Python 令牌不是一个属性访问，那么 prev_token 就是一个函数调用的函数名。
        yield prev_token, not is_attribute_access
        is_attribute_access = False
      else:
        if prev_token:
          # 如果 prev_token 存在，且 prev_token 是一个分隔符，且 prev_token 是 '.'。
          # 这个条件的作用是判断 prev_token 是否是一个属性访问的分隔符。
          # 如果是，就将 is_attribute_access 设置为 True。
          # 表示当前 Python 令牌是一个属性访问。
          is_attribute_access = (
              prev_token.type == tokenize.OP and prev_token.string == '.'
          )
          # 通过 yield 语句将 prev_token 和 is_attribute_access 作为一个元组返回。
          # false 表示 prev_token 不是一个函数调用的函数名。
          yield prev_token, False
      prev_token = token # 将当前 Python 令牌赋值给 prev_token
    # 如果 prev_token 存在，就将 prev_token 和 false 作为一个元组返回。
    # false 表示 prev_token 不是一个函数调用的函数名。
    if prev_token:
      yield prev_token, False
  except Exception as e:
    logging.warning('Failed parsing %s', code)
    raise e

# 这个函数接受一个字符串 code 作为参数，这通常是一段 Python 代码
def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
  """Renames function calls from `source_name` to `target_name`."""
  # 这个函数首先会检查 code 中是否包含 source_name 函数调用
  if source_name not in code:
    return code
  modified_tokens = [] # modified_tokens 的作用是存储修改后的 Python 令牌
  # 这个循环遍历 _yield_token_and_is_call 函数返回的迭代器
  # 这个迭代器的每个元素都是一个元组，元组的第一个元素是一个 tokenize.TokenInfo 对象，第二个元素是一个布尔值，表示这个 TokenInfo 对象是否是一个函数调用。
  for token, is_call in _yield_token_and_is_call(code):
    # 如果是一个函数调用，且函数名是 source_name，就将函数名修改为 target_name。
    if is_call and token.string == source_name:
      # Replace the function name token
      # 使用 tokenize.TokenInfo 类创建一个新的 Python 令牌，表示修改后的函数名。
      modified_token = tokenize.TokenInfo(
          type=token.type,
          string=target_name,
          start=token.start,
          end=token.end,
          line=token.line,
      )
      modified_tokens.append(modified_token)
    else:
      modified_tokens.append(token)
  # 这行代码使用 _untokenize 函数将 modified_tokens 转换为 Python 代码，并返回。
  return _untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
  """Returns the set of all functions called in `code`."""
  return set(token.string for token, is_call in
             _yield_token_and_is_call(code) if is_call)


def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
  """Yields names of functions decorated with `@module.name` in `code`."""
  tree = ast.parse(code)
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
      for decorator in node.decorator_list:
        attribute = None
        if isinstance(decorator, ast.Attribute):
          attribute = decorator
        elif isinstance(decorator, ast.Call):
          attribute = decorator.func
        if (attribute is not None
            and attribute.value.id == module
            and attribute.attr == name):
          yield node.name
