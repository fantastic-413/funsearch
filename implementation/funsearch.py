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

"""A single-threaded implementation of the FunSearch pipeline."""
from collections.abc import Sequence
from typing import Any

from funsearch.implementation import code_manipulation
from funsearch.implementation import config as config_lib
from funsearch.implementation import evaluator
from funsearch.implementation import programs_database
from funsearch.implementation import sampler

# 从specification中提取出要evolve的函数和要run的函数
# specification是一个字符串，是一个python文件的内容,通常包含一些代码。
# 返回类型被定义为 tuple[str, str]，意味着这个函数返回一个包含两个字符串的元组。
def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  # 这行代码使用 code_manipulation.yield_decorated 函数来寻找所有使用 @funsearch.run 装饰器的函数
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  # 接下来的几行检查 run_functions 列表的长度
  # 如果不等于 1，即没有或者有多于一个使用 @funsearch.run 装饰的函数，将抛出一个 ValueError
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]

# main函数的作用是启动一个FunSearch实验
# specification是一个字符串，是一个python文件的内容,通常包含一些代码。
# inputs是一个列表，包含了输入到要run的函数的参数
# config是一个config_lib.Config类型的对象，包含了一些配置信息
def main(specification: str, inputs: Sequence[Any], config: config_lib.Config):
  """Launches a FunSearch experiment."""
  function_to_evolve, function_to_run = _extract_function_names(specification)

  # 这行代码使用 code_manipulation.text_to_program 函数来解析 specification 字符串，返回一个 Program 对象
  # program对象包含了 specification 中的所有信息，包括要evolve的函数和要run的函数
  template = code_manipulation.text_to_program(specification)
  # 这行代码使用 programs_database.ProgramsDatabase 类来创建一个 ProgramsDatabase 对象
  # ProgramsDatabase 类的作用是管理所有的 Function 对象
  database = programs_database.ProgramsDatabase(
      config.programs_database, template, function_to_evolve)

  evaluators = []
  # 这个循环创建了 config.num_evaluators 个 Evaluator 对象
  # Evaluator 类的作用是分析一个函数实现的性能
  for _ in range(config.num_evaluators):
    # 每个 Evaluator 对象都会被添加到 evaluators 列表中
    # Evaluator 类的构造函数接受 5 个参数
    # 1. ProgramsDatabase 对象
    # 2. Program 对象
    # 3. 要evolve的函数名
    # 4. 要run的函数名
    # 5. 要run的函数的参数
    evaluators.append(evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_run,
        inputs,
    ))
  # We send the initial implementation to be analysed by one of the evaluators.
  # 我们将初始实现发送给其中一个评估器进行分析。
  # 这行代码使用 template.get_function 函数来获取要evolve的函数的 Function 对象
  # Function 对象包含了函数的所有信息，包括函数名、参数、返回值类型、文档字符串和函数体
  initial = template.get_function(function_to_evolve).body
  # 这行代码使用 evaluators[0].analyse 函数来分析初始实现
  evaluators[0].analyse(initial, island_id=None, version_generated=None)

  # 这行代码创建了 config.num_samplers 个 Sampler 对象
  # Sampler 类的作用是生成新的实现
  samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt)
              for _ in range(config.num_samplers)]

  # This loop can be executed in parallel on remote sampler machines. As each
  # sampler enters an infinite loop, without parallelization only the first
  # sampler will do any work.
  # 这个循环可以在远程采样器机器上并行执行。由于每个采样器都进入了一个无限循环，因此如果没有并行化，只有第一个采样器才会做任何工作。
  # 这个循环不会结束，除非手动中断程序
  for s in samplers:
    s.sample()
