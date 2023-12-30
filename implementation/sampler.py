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

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import numpy as np

from funsearch.implementation import evaluator
from funsearch.implementation import programs_database

# LLM类的作用是预测给定源代码的续写
class LLM:
  """Language model that predicts continuation of provided source code."""

  # samples_per_prompt的作用是每个prompt获取的独立采样程序续写的数量
  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt

  # _draw_sample函数接收一个prompt，然后返回一个continuation
  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    # 需要给出一个语言模型，然后使用语言模型来生成continuation
    raise NotImplementedError('Must provide a language model.')
  
  # draw_samples函数接收一个prompt，然后返回一个continuation的列表
  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    # 这行代码的作用是调用_draw_sample函数来生成continuation
    # self._samples_per_prompt是每个prompt获取的独立采样程序续写的数量
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

# Sampler类的作用是从数据库中获取prompt，然后使用LLM类来生成continuation，然后将continuation发送给evaluator
# prompt的作用是包含了一些先前的程序，这些程序将被用来生成continuation
# continuation的作用是将prompt中的函数续写成一个完整的程序
# evaluator的作用是评估continuation的质量
class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)

  # sample函数的作用是不断地获取prompt，然后生成continuation，然后将continuation发送给evaluator
  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    while True:
      # get_prompt函数的作用是从数据库中获取prompt
      prompt = self._database.get_prompt()
      # draw_samples函数的作用是使用LLM类来生成continuation
      # prompt.code是prompt中的函数,是一个字符串,包含了一些先前的程序
      samples = self._llm.draw_samples(prompt.code)
      # This loop can be executed in parallel on remote evaluator machines.
      # 这个循环可以在远程评估器机器上并行执行。
      for sample in samples:
        # choice函数的作用是从self._evaluators中随机选择一个evaluator
        chosen_evaluator = np.random.choice(self._evaluators)
        # analyse函数接收sample, prompt.island_id, prompt.version_generated三个参数
        # sample是一个字符串,包含了一个continuation
        # prompt.island_id是一个整数,表示prompt所在的岛屿的id
        chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)
