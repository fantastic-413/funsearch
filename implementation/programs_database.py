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

"""A programs database that implements the evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any

from absl import logging
import numpy as np
import scipy

from funsearch.implementation import code_manipulation
from funsearch.implementation import config as config_lib

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


# _softmax 函数的作用是返回一维有限 `logits` 的 tempered softmax。
# logits 是一个一维数组，包含了每个类别的得分
# temperature 是一个浮点数，表示 softmax 的温度
def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  # 检查 logits 是否包含了非有限值
  if not np.all(np.isfinite(logits)):
    # 如果包含了非有限值，抛出一个 ValueError 异常
    non_finites = set(logits[~np.isfinite(logits)])
    # non_finites 是一个集合，包含了 logits 中的非有限值
    raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  # 检查 logits 的数据类型是否为浮点数
  if not np.issubdtype(logits.dtype, np.floating):
    # 如果不是浮点数，则将 logits 转换成浮点数
    logits = np.array(logits, dtype=np.float32)

  # 使用 scipy.special.softmax 函数将 logits 转换成一个概率分布
  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  # 确保概率之和为1，以防止`np.random.choice`中的错误。
  # 将 result 中的最大值替换成 1 - result 中的其他值之和
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result

# list(scores_per_test.keys())[-1] 的作用是获取 scores_per_test 字典中的最后一个键
# scores_per_test[list(scores_per_test.keys())[-1]] 的作用是获取 scores_per_test 字典中的最后一个值
# 函数作用是将 scores_per_test 字典中的最后一个值返回
# 将每个测试集的得分减少到一个得分。只取最后一个测试集的得分。
def _reduce_score(scores_per_test: ScoresPerTest) -> float:
  """Reduces per-test scores into a single score."""
  return scores_per_test[list(scores_per_test.keys())[-1]]

# 将 scores_per_test 字典中的值按照键的顺序排列，然后转换成一个元组
# 将测试分数表示为规范签名。
def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
  """Represents test scores as a canonical signature."""
  return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


# Prompt 类的作用是存储提示
# 程序数据库生成的提示，将被发送到采样器。
@dataclasses.dataclass(frozen=True)
class Prompt:
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

  Attributes:
    code: The prompt, ending with the header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Identifier of the island that produced the implementations
       included in the prompt. Used to direct the newly generated implementation
       into the same island.
  
  Attributes:
    code: 提示，以要完成的函数的头结尾。
    version_generated: 要完成的函数是 `_v{version_generated}`。
    island_id: 生成提示的岛屿的标识符。用于将新生成的实现引导到同一个岛屿。
  """
  code: str
  version_generated: int
  island_id: int

# 解释 ProgramsDatabase 类的作用是什么？
# ProgramsDatabase 类的作用是存储所有的函数实现，以及它们的得分
class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  # ProgramsDatabase 类的构造函数接受 3 个参数
  # 1. config 是一个 ProgramsDatabaseConfig 类型的对象，包含了一些配置信息
  # 2. template 是一个 Program 对象，包含了要evolve的函数和要run的函数
  # 3. function_to_evolve 是一个字符串，表示要evolve的函数的函数名
  def __init__(
      self,
      config: config_lib.ProgramsDatabaseConfig,
      template: code_manipulation.Program,
      function_to_evolve: str,
  ) -> None:
    self._config: config_lib.ProgramsDatabaseConfig = config
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve

    # Initialize empty islands.
    # 初始化空岛。
    self._islands: list[Island] = [] # self._islands 是一个列表，包含了所有的岛屿
    # 循环 config.num_islands 次，创建 config.num_islands 个 Island 对象，然后添加到 self._islands 列表中
    for _ in range(config.num_islands):
      # Island 类的构造函数接受 5 个参数
      self._islands.append(
          Island(template, function_to_evolve, config.functions_per_prompt,
                 config.cluster_sampling_temperature_init,
                 config.cluster_sampling_temperature_period))
    # self._best_score_per_island 是一个列表，包含了每个岛屿中的程序的最高得分
    self._best_score_per_island: list[float] = (
        [-float('inf')] * config.num_islands)
    # self._best_program_per_island 是一个列表，包含了每个岛屿中的程序的最高得分的程序
    self._best_program_per_island: list[code_manipulation.Function | None] = (
        [None] * config.num_islands)
    # self._best_scores_per_test_per_island 是一个列表，包含了每个岛屿中的程序的最高得分的程序在每个测试集上的得分
    self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
        [None] * config.num_islands)

    # self._last_reset_time 是一个浮点数，表示上一次重置岛屿的时间
    self._last_reset_time: float = time.time()

  # get_prompt 函数的作用随机选择一个岛屿，然后构造一个提示，包含了岛屿中的一组函数实现
  def get_prompt(self) -> Prompt:
    """Returns a prompt containing implementations from one chosen island."""
    # 随机选择一个岛屿
    island_id = np.random.randint(len(self._islands))
    # 调用 Island 类的 get_prompt 函数，构造一个提示，包含了岛屿中的一组函数实现
    # code 是一个字符串，包含了岛屿中的一组函数实现
    # version_generated 是一个整数，表示提示的版本号
    code, version_generated = self._islands[island_id].get_prompt()
    # 构造一个 Prompt 对象，包含了 code、version_generated 和 island_id
    return Prompt(code, version_generated, island_id)

  # _register_program_in_island 函数的作用是将 program 和 scores_per_test 添加到指定的岛屿中
  def _register_program_in_island(
      self,
      program: code_manipulation.Function,
      island_id: int,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the specified island."""
    # self._islands[island_id] 是一个 Island 对象
    # .register_program 函数的作用是将 program 添加到 self._islands[island_id] 中
    self._islands[island_id].register_program(program, scores_per_test)
    # _reduce_score 函数的作用是将 scores_per_test 字典中的最后一个值返回
    score = _reduce_score(scores_per_test)
    # 如果 score 大于 self._best_score_per_island[island_id]，说明 program 的得分比岛屿中的其他程序的得分都要高
    if score > self._best_score_per_island[island_id]:
      # 将 self._best_program_per_island 中的程序替换成 program
      self._best_program_per_island[island_id] = program
      # 将 self._best_scores_per_test_per_island 中的字典替换成 scores_per_test
      self._best_scores_per_test_per_island[island_id] = scores_per_test
      # 将 self._best_score_per_island 中的得分替换成 score
      self._best_score_per_island[island_id] = score
      logging.info('Best score of island %d increased to %s', island_id, score)

  # register_program 函数的作用是将 program 添加到数据库中
  # program 是一个 Function 对象，包含了函数的所有信息，包括函数名、参数、返回值类型、文档字符串和函数体
  # island_id 是一个整数，表示 program 所在的岛屿的 id
  # scores_per_test 是一个字典，包含了 program 在每个测试集上的得分
  def register_program(
      self,
      program: code_manipulation.Function,
      island_id: int | None,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a program on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.
    # 在异步实现中，我们应该考虑在生成提示后重置的岛屿上注册程序的可能性。为了简单起见，我们在这里略过了这一点。
    # 如果 island_id 为 None，说明这是一个在程序开始时添加的程序，所以将它添加到所有的岛屿中
    if island_id is None:
      # This is a program added at the beginning, so adding it to all islands.
      # 这是一个在程序开始时添加的程序，所以将它添加到所有的岛屿中。
      for island_id in range(len(self._islands)):
        self._register_program_in_island(program, island_id, scores_per_test)
    # 如果 island_id 不为 None，说明这是一个新生成的程序，所以将它添加到指定的岛屿中
    else:
      self._register_program_in_island(program, island_id, scores_per_test)

    # Check whether it is time to reset an island.
    # 检查是否是重置岛屿的时候。
    if (time.time() - self._last_reset_time > self._config.reset_period):
      self._last_reset_time = time.time()
      self.reset_islands()

  # 重置岛屿中较弱的一半
  def reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    # We sort best scores after adding minor noise to break ties.
    # 我们在添加微小的噪声后对最佳分数进行排序，以打破平局。
    # np.argsort 函数的作用是返回一个数组，包含了数组中元素的索引，按照元素的值从小到大排序
    # np.random.randn 函数的作用是返回一个数组，包含了从标准正态分布中随机采样的值
    indices_sorted_by_score: np.ndarray = np.argsort(
        self._best_score_per_island +
        np.random.randn(len(self._best_score_per_island)) * 1e-6)
    # num_islands_to_reset 是一个整数，表示要重置的岛屿数量
    num_islands_to_reset = self._config.num_islands // 2
    # reset_islands_ids 是一个列表，包含了要重置的岛屿的 id
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    # keep_islands_ids 是一个列表，包含了要保留的岛屿的 id
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    # 循环 reset_islands_ids 列表，重置岛屿
    for island_id in reset_islands_ids:
      self._islands[island_id] = Island(
          self._template,
          self._function_to_evolve,
          self._config.functions_per_prompt,
          self._config.cluster_sampling_temperature_init,
          self._config.cluster_sampling_temperature_period)
      # 将 self._best_score_per_island 中的得分替换成 -inf
      self._best_score_per_island[island_id] = -float('inf')
      # 随机选择一个 keep_islands_ids 列表中的 id
      founder_island_id = np.random.choice(keep_islands_ids)
      # 并将该岛屿中的最佳程序和最佳分数复制到重置的岛屿中
      founder = self._best_program_per_island[founder_island_id]
      founder_scores = self._best_scores_per_test_per_island[founder_island_id]
      self._register_program_in_island(founder, island_id, founder_scores)


class Island:
  """A sub-population of the programs database."""

  # Island 类的构造函数接受 5 个参数
  # 1. template 是一个 Program 对象，包含了要evolve的函数和要run的函数
  # 2. function_to_evolve 是一个字符串，表示要evolve的函数的函数名
  # 3. functions_per_prompt 是一个整数，表示提示中包含的先前程序数量
  # 4. cluster_sampling_temperature_init 是一个浮点数，表示岛内内聚类的softmax采样的初始温度
  # 5. cluster_sampling_temperature_period 是一个整数，表示聚类采样温度的线性衰减周期
  def __init__(
      self,
      template: code_manipulation.Program,
      function_to_evolve: str,
      functions_per_prompt: int,
      cluster_sampling_temperature_init: float,
      cluster_sampling_temperature_period: int,
  ) -> None:
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    self._functions_per_prompt: int = functions_per_prompt
    self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
    self._cluster_sampling_temperature_period = (
        cluster_sampling_temperature_period)

    self._clusters: dict[Signature, Cluster] = {} # self._clusters 是一个字典，用于存储所有的 Cluster 对象
    self._num_programs: int = 0 # self._num_programs 是一个整数，表示岛屿中的程序数量

  # register_program 函数的作用是将 program 添加到岛屿中
  # program 是一个 Function 对象，包含了函数的所有信息，包括函数名、参数、返回值类型、文档字符串和函数体
  # scores_per_test 是一个字典，包含了 program 在每个测试集上的得分
  def register_program(
      self,
      program: code_manipulation.Function,
      scores_per_test: ScoresPerTest,
  ) -> None:
    # 将程序存储在该岛上，存储在适当的聚类中。
    """Stores a program on this island, in its appropriate cluster."""
    # signature 是一个元组，包含了 program 在每个测试集上的得分
    signature = _get_signature(scores_per_test)
    # 如果 signature 不在 self._clusters 字典中，创建一个新的 Cluster 对象
    if signature not in self._clusters:
      # _reduce_score 函数的作用是将 scores_per_test 字典中的最后一个值返回
      score = _reduce_score(scores_per_test)
      
      self._clusters[signature] = Cluster(score, program)
    # 如果 signature 在 self._clusters 字典中，将 program 添加到对应的 Cluster 对象中
    else:
      self._clusters[signature].register_program(program)
    # 岛屿中的程序数量加 1
    self._num_programs += 1

  # get_prompt 函数的作用是构造一个提示，包含了岛屿中的一组程序
  # 返回值是一个元组，包含了一个字符串和一个整数
  # 字符串是一个提示，包含了岛屿中的一组程序
  # 整数是一个版本号，表示提示的版本号
  def get_prompt(self) -> tuple[str, int]:
    """Constructs a prompt containing functions from this island."""
    # signatures 是一个列表，包含了 self._clusters 字典中的所有键
    signatures = list(self._clusters.keys())
    # 通过 signatures 列表中的键，获取每个 Cluster 对象的得分
    cluster_scores = np.array(
        [self._clusters[signature].score for signature in signatures])

    # Convert scores to probabilities using softmax with temperature schedule.
    # 使用温度调度的softmax将分数转换为概率。
    # period 是一个整数，表示聚类采样温度的线性衰减周期
    period = self._cluster_sampling_temperature_period
    # temperature 是一个浮点数，表示岛内内聚类的softmax采样的温度，temperature的值会随着程序数量的增加而减小
    temperature = self._cluster_sampling_temperature_init * (
        1 - (self._num_programs % period) / period)
    # _softmax 函数的作用是将 cluster_scores 转换成一个概率分布
    probabilities = _softmax(cluster_scores, temperature)

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    # 在实验开始时，当我们有很少的聚类时，将较少的程序放入提示。
    # functions_per_prompt 是一个整数，表示提示中包含的先前程序数量
    functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

    # idx 是一个列表，包含了被选中的 Cluster 对象的索引
    idx = np.random.choice(
        len(signatures), size=functions_per_prompt, p=probabilities)
    # chosen_signatures 是一个列表，包含了被选中的 Cluster 对象的签名
    chosen_signatures = [signatures[i] for i in idx]
    implementations = [] # implementations 是一个列表，包含了被选中的 Cluster 对象中的程序
    scores = [] # scores 是一个列表，包含了被选中的 Cluster 对象的得分
    # 循环 chosen_signatures 列表，遍历每个被选中的 Cluster 对象，获取被选中的 Cluster 对象中的程序和得分
    for signature in chosen_signatures:
      cluster = self._clusters[signature]
      implementations.append(cluster.sample_program())
      scores.append(cluster.score)

    # # indices 是一个列表，包含了 scores 列表中元素的索引，按照元素的值从小到大排序
    indices = np.argsort(scores)
    # sorted_implementations 是一个列表，包含了 implementations 列表中的元素，按照 scores 列表中元素的值从小到大排序
    sorted_implementations = [implementations[i] for i in indices]
    # version_generated 是一个整数，表示提示的版本号，也是表示需要生成的函数的版本号
    version_generated = len(sorted_implementations) + 1
    return self._generate_prompt(sorted_implementations), version_generated

  # 创建一个提示，其中包含一系列函数“实现”。
  def _generate_prompt(
      self,
      implementations: Sequence[code_manipulation.Function]) -> str:
    """Creates a prompt containing a sequence of function `implementations`."""
    # 使用深拷贝，以便我们可以在不改变原始模板的情况下修改其函数。
    implementations = copy.deepcopy(implementations)  # We will mutate these. 

    # Format the names and docstrings of functions to be included in the prompt.
    # 格式化要包含在提示中的函数的名称和文档字符串。
    # versioned_functions 是一个列表，包含了所有的函数
    versioned_functions: list[code_manipulation.Function] = []
    # 循环 implementations 列表，遍历每个 implementation
    for i, implementation in enumerate(implementations):
      # new_function_name 是一个字符串，表示函数的新名称
      new_function_name = f'{self._function_to_evolve}_v{i}'
      # 将函数的名称替换成 new_function_name
      implementation.name = new_function_name
      # Update the docstring for all subsequent functions after `_v0`.
      # 在 `_v0` 之后，更新所有后续函数的文档字符串。
      if i >= 1:
        # 将函数的文档字符串替换成新的文档字符串
        implementation.docstring = (
            f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
      # If the function is recursive, replace calls to itself with its new name.
      # 如果函数是递归的，将对自身的调用也替换为其新名称。
      implementation = code_manipulation.rename_function_calls(
          str(implementation), self._function_to_evolve, new_function_name)
      # 将 implementation 转换成一个 Function 对象，然后添加到 versioned_functions 列表中
      versioned_functions.append(
          code_manipulation.text_to_function(implementation))

    # Create the header of the function to be generated by the LLM.
    # 创建LLM生成的函数的头部。
    # next_version 是一个整数，表示函数的版本号
    next_version = len(implementations)
    # new_function_name 是一个字符串，表示函数的新名称
    new_function_name = f'{self._function_to_evolve}_v{next_version}'
    # dataclasses.replace 函数的作用是利用implementations[-1]创建一个新的 Function 对象
    # name：被设置为变量 new_function_name 的值。
    # body：被设置为空字符串 ''。
    # docstring：被设置为一个新的文档字符串，描述该函数是前一个版本的改进版本。
    header = dataclasses.replace(
        implementations[-1],
        name=new_function_name,
        body='',
        docstring=('Improved version of '
                   f'`{self._function_to_evolve}_v{next_version - 1}`.'),
    )
    # 将 header 转换成一个 Function 对象，然后添加到 versioned_functions 列表中
    versioned_functions.append(header)

    # Replace functions in the template with the list constructed here.
    # 用这里构造的列表替换模板中的函数，作为提示。
    prompt = dataclasses.replace(self._template, functions=versioned_functions)
    # 将 prompt 转换成一个字符串，然后返回
    return str(prompt)

# Cluster 类的作用是存储一组具有相同签名的程序
# 一个岛屿上具有相同签名的程序被存储在一个 Cluster 对象中
class Cluster:
  """A cluster of programs on the same island and with the same Signature."""

  # Cluster 类的构造函数接受 2 个参数
  # 1. score 是一个浮点数，表示该 Cluster 的得分
  # 2. implementation 是一个 Function 对象，包含了函数的所有信息，包括函数名、参数、返回值类型、文档字符串和函数体
  def __init__(self, score: float, implementation: code_manipulation.Function):
    self._score = score
    self._programs: list[code_manipulation.Function] = [implementation]
    self._lengths: list[int] = [len(str(implementation))] # self._lengths 是一个列表，包含了每个程序的长度

  @property
  def score(self) -> float:
    """Reduced score of the signature that this cluster represents."""
    return self._score

  # register_program 函数的作用是将 program 添加到该 Cluster 中
  def register_program(self, program: code_manipulation.Function) -> None:
    """Adds `program` to the cluster."""
    self._programs.append(program)
    self._lengths.append(len(str(program)))

  # sample_program 函数的作用是从该 Cluster 中随机选择一个程序
  # 该函数会根据程序的长度来选择程序，长度越短的程序被选中的概率越大
  def sample_program(self) -> code_manipulation.Function:
    """Samples a program, giving higher probability to shorther programs."""
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
        max(self._lengths) + 1e-6)
    # _softmax 函数的作用是将 normalized_lengths 转换成一个概率分布
    probabilities = _softmax(-normalized_lengths, temperature=1.0)
    # np.random.choice 函数的作用是从 self._programs 列表中随机选择一个程序
    # p=probabilities 表示每个程序被选中的概率
    return np.random.choice(self._programs, p=probabilities)
