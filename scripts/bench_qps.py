"""F32 压测：并发能到多少、延迟长什么样。

## 为什么要分三档，而不是报一个数

"你这服务 QPS 多少" 这个问题**不能用一个数字回答**，因为一次问答的耗时里
混着三种量级完全不同的东西：

| 档 | 真检索 | 真模型 | 测的是 | 典型量级 |
|---|---|---|---|---|
| `mock` | 假 | 假 | **服务壳本身**：HTTP + 事件序列化 + 会话锁 | 毫秒 |
| `nollm` | 真 | 假 | 壳 + **检索链路**（编码、融合、精排） | 几百毫秒 |
| `real` | 真 | 真 | 端到端，**含模型网络往返** | 几秒到十几秒 |

只报 `real` 那个数，读者会以为瓶颈在服务层（于是就有人去优化 FastAPI 的路由）；
只报 `mock` 那个数，又会被当成"你系统能扛这么多请求"。
**两个数都有信息量，混在一起说就都没有了。**

默认只跑 `mock`（免费、不需要 ml extra、十几秒跑完）。
`nollm` 和 `real` 要显式指定——`real` 会花钱。

    python scripts/bench_qps.py                        # 只跑 mock
    python scripts/bench_qps.py --tier nollm           # 加真检索（要 ml extra）
    python scripts/bench_qps.py --tier real --tier-n 6 # 真调 API，**要钱**

    # 只要几个并发、多跑几轮（验证某一条结论时用，省时间）
    python scripts/bench_qps.py --conc 1 --conc 32 --repeat 6

    # 诊断：并发掉吞吐，是服务的问题还是线程抢锁的问题？先生成对照组
    python scripts/bench_qps.py --probe
    # 再直接把 app 的线程池压到 1 个线程，看吞吐回来没有
    python scripts/bench_qps.py --conc 1 --conc 32 --repeat 6 --threads 1 --out /tmp/x.md

## 两个场景，回答的是两个不同的问题

- `fanout`：每个请求**用自己的 session**。
  问的是"服务能并行处理多少互不相关的人"。
- `same`：所有请求打**同一个 session**。
  问的是"同一个人连发几条会怎样"。

这两个数的差距不是 bug，是**设计**：`SessionStore` 给每个会话一把
`asyncio.Lock`，同一个会话的请求本来就该排队——不排队就会出现
"两个请求各自读到 version=7、各自写回，后写的盖掉先写的"。

## ⚠️ 这个脚本最花心思的地方：不让自己测出假的结论

第一版压测跑出一张**会骗人的表**，排查过程值得完整记下来，
因为它是这个项目里最容易犯的一类错：**把机器的抖动当成系统的性质。**

1. 同一个配置（mock / fanout / 并发 32 / 96 个请求），单独跑时 98 QPS，
   放进整跑里只有 54 QPS。差一倍。
2. 顺着"越跑越慢"查下去：先怀疑会话表越堆越大（`OrderedDict` + LRU，
   取会话 O(1)，**不是它**）；把会话表清空再 `gc`，QPS 一点没回来
   （55.5 → 57.0，**不是内存**）。
3. 拿一段**与项目毫无关系的纯 Python 计算**（16 线程各烧定长循环）连跑 15 秒：
   头 1.6 秒每轮约 72ms，第 5 秒变成约 122ms，之后稳定在 1.4~1.6 倍。
   **这台机器（云主机，有 CPU 配额）持续满载几秒后就进"慢档"，
   而且歇 8 秒也回不来。**
4. 当时的"决定性一步"：**把并发 32 提到最前面跑**（此刻机器还快），
   得 98.5 QPS，和并发 1 的 89.2 差不多。于是结论写成了
   "并发不掉吞吐，掉的是机器状态"。

**第 4 步那个结论后来被推翻了，而且推翻它的正是上面这套防坑机制。**

把 harness 修好之后（每配置 6 轮取最快、偶数轮把配置顺序整个倒过来），
只跑并发 1 和并发 32 这两个点，重新量：

| `mock` 档 | 并发 1 | 并发 32 |
|---|---:|---:|
| `fanout`（各自独立会话） | 101.6 | **57.1** |
| `same`（全是同一个会话） | 80.3 | 77.6 |

（6 轮取最快；那两条最快的标尺分别是 1.00x 和 1.26x，说明**两次都跑在快机器上**，
不是机器慢造成的。）

**并发 32 确实只剩并发 1 的一半多一点**——这个差不可能是顺序偏差，
因为顺序已经倒过来跑过了。那之前那个 98.5 是什么？
**单次采样。** 重复六次就再也测不出来。
一句话：**一个样本打不过六个样本**，哪怕那一个样本当时看着像"决定性证据"。
（顺带一个教训：越是"一步定案"的实验，越要多想一层——它把结论带向的方向，
恰好是作者本来希望的方向。）

### 那到底为什么掉？（两个实验，不是一个）

**实验一：对照组——多线程跑纯 Python 亏不亏（`--probe`）。**

同一份总计算量，①一个线程串行做完，②丢给 20 个线程同时做。
总工作量一样，所以理想情况下两者一样快，比值应该是 1.00。实测：

| 计算形状 | 比值 ②/① |
|---|---:|
| 紧凑整数循环（只碰寄存器） | **1.00x** |
| 大量小函数调用 | 0.96x |
| **拼命分配内存（拼字符串、建字典）** | **1.57x** |

**形状不同，结论完全不同。** 所以"多线程跑 Python 亏不亏"这个问题本身是错的，
得看那段代码在不在堆上反复申请释放。而**真实业务代码就是最后那一种**
（拼字符串、建字典、建对象、序列化 JSON），亏 1.57x——和 app 里掉的那 1.8x 同一个量级。
机制说白了：GIL 同一时刻只让一个线程跑 Python，被踢下来的线程要把手上的东西
放下、换下一个上来；只要线程在不停地申请/释放内存，这个"换手"就特别贵。

**实验二：直接把 app 的线程池压到 1 个线程（`--threads 1`）。**

上面那个是"机器性格"的证据，这个是"就在这个服务上"的证据。
把 `asyncio.to_thread` 的池子从 20 压到 **1**，同一个 mock 档重跑：

| `mock` / `fanout` | 线程池 20（默认） | 线程池 1 |
|---|---:|---:|
| 并发 1 | 101.6 | 99.6 |
| 并发 32 | **57.1** | **90.2** |

**池子从 20 压到 1，并发 32 的吞吐从 57 涨到 90**（p50 也从 539ms 降到 337ms）。
线程少了反而快——因为活是纯 Python、本来就被 GIL 串着，多出来的线程
只是在抢锁和上下文切换上白花功夫。

**所以修法只有一个方向：加进程**（`uvicorn --workers N`），不是在这个进程里加并发。
本项目没做多进程部署，所以这个数不在这张表里。

**顺带解释了 `same` 为什么反而更好。** 表里 `same` 在并发 32 上是 77.6，
比 `fanout` 的 57.1 高。原因不是"锁好"，而是**那把会话锁不小心把并发压回了 1**
（同一个会话的请求本来就该排队），于是正好绕开了上面这个线程争抢。
**它是被排队救的，不是被锁救的。** 这一条值得单独记：一个为了"正确性"
（防止两个请求互相盖写）加的锁，顺带带来了一点点性能好处，
但那是副作用，不是设计目标——**不能因为"看着更快"就以为排队是好事**。

### 怎么对付机器的抖动

试过、并且**否掉**的办法：每个配置前面量一把标尺，按倍率折算 QPS。
否掉的理由很实在——标尺自己一次只有 70~130ms，噪声约 ±10%，而这个倍率
要**乘到 QPS 上**，噪声跟着放大。实测折算完出现过 137 QPS 这种
**比冷机还高**的怪数。**用一把不准的尺子去量，只会得到更不准的数。**

现在用的是笨办法，但它对：**每个配置跑若干轮，取最快的那轮**——
轮数**按档定**（`mock`/`nollm` 各 3 轮，`real` 1 轮，见 `TIER_DEFAULT_REPEAT`），
而不是整场一个数。踩过：写成「这次只要跑了 `real` 就全场只跑 1 轮」，
连免费档的 best-of-K 也一起砍了。
它成立只需要一个前提——**机器只会比它自己的能力强项慢，不会更快**。
被邻居抢占、被 CPU 配额压住，都只让数字变小；所以"最快的一遍"
就是这台机器能力的下界估计。这个前提不需要标尺准，
而且**顺序偏差被摊平了**：第 2 轮的配置顺序整个倒过来跑，
"谁先跑谁占便宜"不会固定落在某个配置头上。

标尺没有丢，它降级成了一个**诊断列**（`标尺ms`）：它告诉你量这个配置时
机器比全程最快时慢了几倍。数不好看的时候先看这一列，别急着改代码。

⚠️ **这套办法能救什么、不能救什么，得分清：** 它救的是"机器时快时慢"——
同一个配置多跑几轮，总有几次落在快的时段，取最快就绕过去了。
它**救不了**"本次运行从头到尾都慢"：比如 `nollm` 档开头那 20 秒模型加载，
本身就满载，会把机器压进慢档，于是那一档 3 轮全是慢的。
所以**绝对数字不可移植**（换台安静的机器会明显更好），
能带走的只有形状——**吞吐随并发怎么变**。

## 每个配置一个干净的会话表

每次测量都新建一个 app（重的模型加载由 `_heavy_parts` 拆出来、每档只做一次，
所以重建 app 本身很便宜）。这样配置之间没有"上一个配置留在会话表里的东西"
这个额外变量——上面第 2 步已经证明它不是主因，但**能少一个变量就少一个**。

## 这个压测**测不到**什么（必须说清楚）

1. **没有经过 uvicorn 的网络栈。** 请求直接喂给 ASGI app
   （`httpx.ASGITransport`），没有 TCP、没有 socket 读写、没有 uvicorn
   自己的协议解析。所以 `mock` 档是服务层吞吐的**上界**。
2. **单进程、单事件循环。** 所有数字都是"一个 Python 进程"的。
   要提吞吐得加进程（`uvicorn --workers N`），那不属于这里测的。
3. **图跑在线程池里**，池子大小是 `min(32, cpu+4)`。
   并发超过它之后，多出来的请求在**排队等线程**，不是等 CPU。
4. **`mock` 档会把线程争抢这件事放大。** 它的活是假的、零 IO，
   每个请求几乎一直在算 Python 字节码——这正是 GIL 最难受的形状。
   现实里每个请求大半时间在等模型网络往返，线程大多是闲的，
   抢 GIL 远没这么严重。**所以 `mock` 档掉的吞吐，别当成线上会掉的量。**
   真要看加并发值不值，看 `nollm` / `real` 那两档（它们没掉）。
5. **绝对数字不可移植。** 换台机器（尤其换掉这台云主机的 CPU 配额）
   这些数都会变。能带走的是**形状**——吞吐随并发怎么变、加并发换来了什么。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # 为了让 `tests.fakes` 能被 import，理由见下

from juris_agent.api.app import create_app  # noqa: E402
from juris_agent.api.runtime import build_runtime  # noqa: E402
from juris_agent.config import get_settings  # noqa: E402

# ⚠️ **假件直接从 `tests/` 里借，不在这个脚本里另写一份。**
#
# 看着有点别扭（脚本依赖测试目录），但反过来更糟：另写一份假的检索器，
# 两份就会慢慢长歪——"测试里那个假件"和"压测里那个假件"行为不同，
# 于是测试全绿而压测数字对不上，没人能解释为什么。
# 借过来用，**保证压测测的就是测试测的那个壳**。改动一处，两边一起变。
from tests.fakes import FakeRetriever, tool_call  # noqa: E402

# --------------------------------------------------------------------------- #
# 机器标尺（诊断用，不再拿它折算 QPS，理由见文件头）
# --------------------------------------------------------------------------- #

# 标尺用几个线程。取一个跟测试并发数同量级的值（16），而不是跟着 `conc` 变——
# 标尺必须是**固定的**，否则它就不是标尺了。
YARDSTICK_THREADS = 16

# 每个线程烧多少圈。只影响标尺的绝对耗时（这里约 70ms），不影响倍率。
# **别调太小**：太小会被线程启动/调度开销盖住，倍率全是噪声。
YARDSTICK_LOOPS = 120_000

# 标尺连量几轮取最小值。理由：这里问的是"这台机器**能力**是多少"，
# 答案天然是"它最快的那一次"——平均值里混的是调度抖动和邻居抢占。
YARDSTICK_ROUNDS = 5


def _burn(n: int = YARDSTICK_LOOPS) -> int:
    """一段纯 Python 计算，**故意和项目没有任何关系**。

    为什么不用"跑一次检索"或者"跑一轮问答"当标尺？因为那样标尺本身
    就依赖被测代码——被测代码一改，标尺跟着变，"这台机器快了还是慢了"
    和"我的改动生效了没有"就再也分不开了。
    要量机器，就得用**不随项目变**的东西。
    """
    s = 0
    for i in range(n):
        s += i * i
    return s


# 标尺**自己带一个线程池**，不走事件循环的默认池。
#
# 两个理由：
# ① 标尺和被测 app 抢同一个池子的话，"这台机器多快"这个数里就掺进了
#    "池子被占了多少"，两件事混在一个数里，谁也别想解释。
# ② `--threads` 那个开关（把 app 的池子压到 1 个线程，用来验证
#    "并发掉吞吐到底是不是线程抢锁造成的"）会顺手把标尺也压死，
#    诊断列跟着一起失真。分开池子，这个开关就只作用于被测对象。
_YARD_EXEC = ThreadPoolExecutor(max_workers=YARDSTICK_THREADS,
                                thread_name_prefix="yardstick")


async def yardstick(rounds: int = YARDSTICK_ROUNDS) -> float:
    """量一把"这台机器现在最快能多快"，单位毫秒。**越小越快。**

    同一段固定计算，快的时候约 70ms，被 CPU 配额压住时约 115~130ms。
    **它只用于诊断列**，不参与 QPS 折算——理由见文件头那节
    "试过、并且否掉的办法"。
    """
    loop = asyncio.get_running_loop()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        await asyncio.gather(
            *(loop.run_in_executor(_YARD_EXEC, _burn) for _ in range(YARDSTICK_THREADS))
        )
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _burn_alloc(n: int) -> int:
    """另一段纯 Python 计算，**故意和 `_burn` 长得不一样：这段不停地分配内存**。

    为什么非要多写一段？因为只用 `_burn`（紧凑整数循环）会得出**错误结论**——
    实测那段循环 20 线程并行和单线程串行一样快（1.00x），
    于是"多线程不亏"这个结论就成立了，而它和 app 里的现象（并发一高掉吞吐）矛盾。

    差别在**分配**：紧凑整数循环只碰寄存器里的几个整数，不碰堆，
    线程之间几乎没有共享的写；而真实代码（拼字符串、建字典、建 pydantic 对象、
    序列化 JSON）一直在堆上申请/释放，解释器要在所有线程之间协调。

    实测（同一份总计算量，20 线程并行 ÷ 单线程串行）：

    | 计算形状 | 比值 |
    |---|---|
    | 紧凑整数循环 | 1.00x（不亏） |
    | 大量小函数调用 | 0.96x（不亏） |
    | **拼命分配内存** | **1.72x（亏 72%）** |

    **所以"多线程跑纯 Python 亏不亏"这个问题问错了**——得看那段代码在干什么。
    这也就是为什么这个探针要跑两种形状，而不是一种。
    """
    d: dict[int, str] = {}
    out: list[int] = []
    for i in range(n):
        d[i & 31] = f"{i}:{i * i}"
        out.append(len(d))
    return len(out) + len(d)


# （形状名, 函数, 每个线程烧多少圈）。圈数是按"单份计算大约几百毫秒"配的，
# 太小会被线程启动开销盖住，太大就让这个探针从"几秒"变成"几分钟"。
PROBE_SHAPES = [
    ("紧凑整数循环（不碰堆）", _burn, 400_000),
    ("拼命分配内存（拼字符串/建字典）", _burn_alloc, 200_000),
]


async def thread_probe(*, units: int = 20, rounds: int = 4) -> list[dict]:
    """对照组：**同一份纯 Python 计算量，串行跑一遍 vs 丢给 20 个线程并行跑一遍。**

    这个探针回答的是压测里最要紧的一个"为什么"：

    > `mock` 档并发 1 能跑 100 QPS，并发 32 只剩 57 QPS。
    > 这是**服务**的问题，还是**这台机器 + GIL** 的问题？

    做法和被测代码毫无关系（就是一段自造的计算），所以它量的是**这台机器的性格**，
    不是项目的性质：

    - `serial_ms`：一个线程把 `units` 份计算一份接一份做完。
    - `parallel_ms`：`asyncio.gather` 把这 `units` 份**同时**丢进
      `asyncio.to_thread` 默认线程池（池子大小就是 `units`，正好塞满）。

    **两份的"总工作量"完全一样**，所以理想情况下两者应该一样快，
    比值 `parallel_ms / serial_ms` ≈ **1.0**。

    - 比值 ≈ 1.0 → 线程之间不互相拖累。那并发一高吞吐就掉，就别赖机器。
    - 比值明显 > 1.0 → **这个形状的活，多线程跑就是更慢**。
      那么"加并发反而掉吞吐"就有了机器层面的解释，
      而修法只有一个方向：**加进程**，不是加线程。

    每一轮里串行、并行**交替**做（而不是"串行全跑完再跑并行"），
    这样机器如果在慢慢变慢，两边的吃亏程度一样，比值仍然可比。
    这跟压测主体里"第 2 轮把配置顺序倒过来"是同一个道理。
    """
    out: list[dict] = []
    for name, fn, loops in PROBE_SHAPES:
        serial: list[float] = []
        parallel: list[float] = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(units):
                fn(loops)
            serial.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            await asyncio.gather(*(asyncio.to_thread(fn, loops) for _ in range(units)))
            parallel.append(time.perf_counter() - t0)

        s, p = min(serial), min(parallel)
        out.append({"name": name, "serial_ms": s * 1000, "parallel_ms": p * 1000,
                    "ratio": p / s, "units": units, "loops": loops, "rounds": rounds})
    return out


def print_thread_probe(prs: list[dict], pool: int) -> None:
    """把对照组的结果打成人能读的样子。**只打印，不写进报告**——
    它是一次性诊断，每次跑出来的绝对值都不同，塞进报告会很快过期。
    结论会被写进报告，数字不会。
    """
    print("=" * 78)
    print("对照组：多线程跑纯 Python 到底亏不亏（和本项目毫无关系的一段计算）")
    print("=" * 78)
    print(f"  做法：同一份总计算量，① 单线程串行做完，② 丢给 {pool} 个线程同时做。")
    print("  两份的总工作量一模一样，所以理想情况下两者一样快，比值应该是 1.00。")
    print()
    for pr in prs:
        print(f"  [{pr['name']}]  每份 {pr['loops']:,} 圈 × {pr['units']} 份 × {pr['rounds']} 轮取最快")
        print(f"      ① 单线程串行 : {pr['serial_ms']:7.0f} ms")
        print(f"      ② 多线程并行 : {pr['parallel_ms']:7.0f} ms")
        print(f"      比值 ②/①    : **{pr['ratio']:.2f}x**")
    print()
    worst = max(prs, key=lambda p: p["ratio"])
    if worst["ratio"] < 1.10:
        print("  → 两种形状都不亏。并发一高吞吐就掉的话，别赖机器，去查服务代码。")
    else:
        print(f"  → **形状不同，结论不同**：`{worst['name']}` 亏 {worst['ratio']:.2f}x，")
        print("    另一种几乎不亏。所以「多线程跑 Python 亏不亏」这个问题本身问错了——")
        print("    要看那段代码在不在堆上反复申请释放。**真实业务代码是后一种**，")
        print("    这就是本进程内加并发会掉吞吐的原因。修法：加进程，不是加线程。")


# --------------------------------------------------------------------------- #
# 问题集
# --------------------------------------------------------------------------- #

# 混了四种形态，因为一次问答走的路不一样，耗时也不同：
# 纯法条问答（一次生成）、计算题（工具循环 → 两三次模型调用）、
# 超纲（**一次模型调用都没有**，规则路由直接拒）、闲聊。
#
# ⚠️ **如果只用同一种问题压测，得到的 QPS 是那一种问题的 QPS。**
# 报告里必须写清这个数是哪几种问题混出来的。
QUESTIONS = [
    "经济补偿怎么算",
    "试用期最长可以约定多久",
    "加班费怎么算",
    "竞业限制没约定补偿怎么办",
    "被违法辞退能拿多少赔偿",
    "你好",
    "帮我写一首关于秋天的诗",
    "劳动合同到期不续签有补偿吗",
]

_FILLER = "《劳动合同法》第四十七条 经济补偿按劳动者在本单位工作的年限，按每满一年支付一个月工资的标准支付。"


# --------------------------------------------------------------------------- #
# 造 app
# --------------------------------------------------------------------------- #


def _heavy_parts(tier: str) -> tuple:
    """这一档里"贵"的两样东西：检索器和模型。**每档只造一次，而且全部提前造。**

    为什么拆出来：每次测量都要一个**崭新的 app**（干净的会话表），
    而 `make_app` 里真正贵的是加载 bge 那两个模型（约 20 秒）。

    ⚠️ 还有一个更重要的理由：**这 20 秒的满载会把机器推进"慢档"**。
    所以零件的造法很讲究——**按档惰性造，而且只在这一档开测之前造**
    （见 `main_async` 里 `parts[tier] = _heavy_parts(tier)` 那一行）：
    这样那 20 秒的满载夹在"上一档测完"和"这一档开测"之间，
    只拖慢它自己这一档，不会把别的档也拖下水。

    踩过的反面做法：一上来就把所有档的零件全造完——于是**连第一档
    mock 都是在慢机器上跑的**（标尺列从第一条就是 1.2~1.5x），
    整张表白白矮一截。
    """
    s = get_settings()
    if tier == "mock":
        return s, FakeRetriever(), RepeatingLLM()
    if tier == "nollm":
        from juris_agent.retrieval.build import build_default_retriever

        return s, build_default_retriever(settings=s), RepeatingLLM()
    if tier == "real":
        from juris_agent.llm import LLMClient
        from juris_agent.retrieval.build import build_default_retriever

        # ⚠️ **这里必须自己把 `LLMClient` 造出来，不能图省事传 `llm=None`
        # 然后指望 `build_runtime` 去造。** 踩过这个坑，症状特别隐蔽：
        #
        # `make_app` 里是先 `make_runner(llm=llm)` 再 `build_runtime(llm=llm)`。
        # 传 `None` 的话，**图是在"没有模型"的情况下建起来的**；
        # 等 `build_runtime` 发现 `llm is None` 去造一个真客户端时，
        # `runner` 已经不为 `None` 了（第 155 行那个 `runner is None` 不成立），
        # 于是它造出来的客户端**根本没人用**。
        #
        # 结果：这一档跑出来的 p50 是 159ms——**真调 API 不可能这么快**
        # （光网络往返就一秒起）。是那个数字让这个 bug 露的马脚。
        # 教训：**"降级"和"没接上"看起来一模一样，都会安安静静跑完、
        # 返回 200，只有一个不可信的数字能戳穿它。**
        return s, build_default_retriever(settings=s), LLMClient()
    raise ValueError(tier)


def make_app(tier: str, parts: tuple):
    """用（可能复用的）重型部件造一个**全新的** app：会话表、执行器、图状态都是新的。

    **三档的区别只在"检索器"和"模型"用真的还是假的**，
    HTTP 层、会话层、事件层、图——三档完全相同。
    这样三档之间的差才是"检索"和"模型"贡献的，而不是别的什么。
    """
    s, retriever, llm = parts

    from juris_agent.agent.graph import make_runner

    runner = make_runner(retriever=retriever, llm=llm, settings=s)
    rt = build_runtime(s, runner=runner, retriever=retriever, llm=llm)
    if rt.problems:
        print(f"[FATAL] 装配失败：{rt.problems}", file=sys.stderr)
        raise SystemExit(2)
    return create_app(rt, serve_web=False)


class RepeatingLLM:
    """一个**每次都能重来一遍**的假模型（无状态、按消息内容决定回什么）。

    ⚠️ **不能直接用 `ScriptedLLM`**：它的剧本是**一次性**的，用完回空正文。
    单请求测试里没问题（每次新建一个），但压测要跑几百个请求，
    共享一个实例的话第二个请求开始就拿到空正文了——
    于是"越到后面越快"（少一次模型调用），**曲线会好看得离谱**。
    这种"压测工具自己造成的偏差"不报错、不失败，只会给出一张假的表，
    是压测里最难发现的一类错误。

    判断依据是**消息里有没有工具结果**（`role == "tool"`），
    不是"第几次调用"——后者要维护一个跨线程的计数器，
    在并发下既不准又难解释。
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kw):
        from juris_agent.llm import LLMResponse, Usage

        self.calls += 1
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if kw.get("tools") and not has_tool_result:
            return LLMResponse(
                text="",
                model="fake",
                usage=Usage(total_tokens=10),
                tool_calls=[tool_call(
                    f"c{self.calls}", "calc_severance",
                    {"cause": "illegal_dismissal", "years": 3, "avg_monthly_wage": 12000},
                )],
            )
        return LLMResponse(text=_FILLER, model="fake", usage=Usage(total_tokens=10))


# --------------------------------------------------------------------------- #
# 跑
# --------------------------------------------------------------------------- #


def _fmt_qps(x: float) -> str:
    """QPS 的显示格式。**小数值不能用 `.1f`。**

    踩过：`real` 档一条请求十几秒，QPS 是 0.05 这种量级，
    用 `.1f` 全被格式化成 `0.0`——**一眼看过去像"服务死了"**，
    而其实只是单位选错了。低于 1 就多给两位小数。
    """
    if x != x:  # NaN
        return "—"
    return f"{x:.1f}" if x >= 1 else f"{x:.2f}"


class Result:
    """一个配置**一次测量**的结果。

    `yard_ms` 是开跑之前那把标尺的耗时（诊断用），**不是**折算系数——
    折算那条路已经被否掉了，理由见文件头。
    """

    def __init__(self, tier: str, scenario: str, conc: int, lat: list[float],
                 wall: float, errors: int, n: int) -> None:
        self.tier, self.scenario, self.conc = tier, scenario, conc
        self.lat, self.wall, self.errors, self.n = sorted(lat), wall, errors, n
        self.yard_ms = float("nan")
        # 这个配置跑了几轮。**是按档定的**（见 `TIER_DEFAULT_REPEAT`），
        # 不是整场一个数——理由写在 `main_async` 里。
        self.repeat = 1

    def q(self, p: float) -> float:
        if not self.lat:
            return float("nan")
        i = min(len(self.lat) - 1, int(round(p * (len(self.lat) - 1))))
        return self.lat[i]

    @property
    def qps(self) -> float:
        return self.n / self.wall if self.wall > 0 else float("nan")

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.tier, self.scenario, self.conc)

    def row(self, ref_ms: float = float("nan")) -> str:
        """一行 markdown。`ref_ms` 是全程最快的标尺，用来把 `yard_ms` 表示成倍率。"""
        if ref_ms == ref_ms and self.yard_ms == self.yard_ms and ref_ms > 0:
            rel_s = f"{self.yard_ms / ref_ms:.2f}x"
        else:
            rel_s = "—"
        return (f"| {self.tier} | {self.scenario} | {self.conc} | {self.repeat} | {self.n} | "
                f"{self.q(0.50) * 1000:.0f} | {self.q(0.95) * 1000:.0f} | "
                f"{self.q(0.99) * 1000:.0f} | **{_fmt_qps(self.qps)}** | {rel_s} | {self.errors} |")


async def one_request(client: httpx.AsyncClient, q: str, session_id: str) -> tuple[bool, float]:
    """发一轮问答，返回 (成不成功, 耗时秒)。

    打到 `/api/chat/stream`——**浏览器走的就是这条路**，
    压测当然也该压它，而不是压那条只有脚本会用的非流式接口。

    `client.post()` 会把整个响应读完之后才返回，所以量到的是
    "这一轮从头到尾跑完"的耗时。用 `stream()` 手动读也可以，
    但那会多引入"读到第几个 chunk 算完"的定义问题，而这里不需要那个粒度。
    """
    t0 = time.perf_counter()
    try:
        r = await client.post(
            "/api/chat/stream",
            json={"question": q, "session_id": session_id, "stream_llm": False},
        )
        ok = r.status_code == 200 and "event: done" in r.text
    except Exception:  # noqa: BLE001
        ok = False
    return ok, time.perf_counter() - t0


async def run_config(app, tier: str, scenario: str, conc: int, n: int, warmup: int) -> Result:
    """跑一组：`warmup` 个不计时的请求 + `n` 个计时的请求。

    ⚠️ **预热段和被统计段是分开计时的，不是"总时间减掉几个"。**
    首次请求里含 jieba 词典、线程池起线程、连接池建立——
    把它们混进统计，p99 就变成"第一个请求的耗时"，
    而那个数随机器状态变，不是服务的性质。
    分开计时的好处是**分母也干净**：QPS = n / 被统计那一段的墙钟时间。
    """
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://bench",
        timeout=300.0,
    )
    fixed_sid = f"bench-{tier}-{scenario}-{conc}"
    sem = asyncio.Semaphore(conc)
    lat: list[float] = []
    errors = 0

    def sid_for(i: int) -> str:
        # `fanout` 每个请求一个新会话（这才是真的并行）；
        # `same` 全部打同一个会话（会被那把锁串起来，见文件头）。
        return fixed_sid if scenario == "same" else f"{fixed_sid}-{i}"

    async def worker(i: int, collect: bool) -> None:
        nonlocal errors
        async with sem:
            ok, dt = await one_request(client, QUESTIONS[i % len(QUESTIONS)], sid_for(i))
            if not ok:
                errors += 1
            elif collect:
                lat.append(dt)

    try:
        if warmup:
            await asyncio.gather(*(worker(i, False) for i in range(warmup)))
        wall = 0.0
        if n:
            t0 = time.perf_counter()
            await asyncio.gather(*(worker(warmup + i, True) for i in range(n)))
            wall = time.perf_counter() - t0
    finally:
        await client.aclose()
    return Result(tier, scenario, conc, lat, wall, errors, n)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


# `real` 的数小不是因为它在系统里更"轻"，而是因为**它在花钱、而且每条要十几秒**：
# 每个配置 2 个预热 + 4 个计时，4 个配置一共 24 次真调。想多要点样本就传 `--tier-n`，
# 但先想清楚要花多少——顺便，样本这么少的时候 **p95 基本没有意义**，
# 别拿它当 SLO 用。
TIER_DEFAULT_N = {"mock": 96, "nollm": 32, "real": 4}
TIER_DEFAULT_REPEAT = {"mock": 3, "nollm": 3, "real": 1}
CONCURRENCIES = {"mock": [1, 4, 16, 32], "nollm": [1, 4, 8], "real": [1, 2]}


def concs_for(tier: str, args) -> list[int]:
    """这一档要跑哪几个并发。`--conc` 给了就用它（**但必须在档的默认表里**）。

    为什么要 `--conc`：想专门验证一条结论时（比如"并发 1 和并发 32
    到底谁快"），把整张表跑一遍是浪费——`nollm` 跑一轮好几分钟，
    `real` 还花钱。有了这个开关就能只跑关心那两个点，多跑几轮。
    这也是 `--repeat` 存在的理由，两者经常一起用。

    为什么**不允许**传表里没有的数字：那样跑出来的行没法进报告
    （报告的行是按 `CONCURRENCIES` 排的），而且"随手补一个 8"
    会让两次报告的行集不一样，**两份报告就没法对看了**。
    """
    if not args.conc:
        return CONCURRENCIES[tier]
    allowed = CONCURRENCIES[tier]
    bad = [c for c in args.conc if c not in allowed]
    if bad:
        raise SystemExit(f"--conc {bad} 不在 {tier} 档的取值里（可选：{allowed}）")
    return [c for c in allowed if c in args.conc]  # 按默认顺序，不让命令行顺序影响输出顺序


async def main_async(args) -> int:
    tp = min(32, (os.cpu_count() or 1) + 4)

    if args.threads:
        # 换掉事件循环的默认执行器。**必须在这里换，不能在 `run_config` 里换**——
        # `asyncio.to_thread` 每次调用都去问当前事件循环要默认执行器，
        # 所以换在循环上就够了；换在别处（比如自己传 executor）会漏掉
        # 图内部那些同样走 `to_thread` 的地方，测出来的就不是真的"池子小"。
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=args.threads, thread_name_prefix="app")
        )

    print("=" * 78)
    print("F32 压测")
    print("=" * 78)
    print(f"  档位          : {'、'.join(args.tier)}")
    pool_s = f"{args.threads}（--threads 指定的，诊断模式）" if args.threads else \
        f"{tp}（`asyncio.to_thread` 的默认上限）"
    print(f"  事件循环线程池: {pool_s}")
    if args.threads:
        print("  ⚠️ 这是诊断跑法，**不要拿这张表的 QPS 当服务能力**——"
              "池子被压小了，测的是「线程少了会怎样」。")
    print(f"  CPU 核心      : {os.cpu_count()}")
    print("  说明          : 请求直接喂 ASGI app，**没走 uvicorn 的网络栈**；单进程")

    parts: dict[str, tuple] = {}
    best: dict[tuple[str, str, int], Result] = {}
    yards: list[float] = []
    for tier in args.tier:
        n = args.tier_n or TIER_DEFAULT_N[tier]
        # ⚠️ **轮数是"按档"定的，不是"按整场"定的。**
        # 踩过：写成"这次只要跑了 real，就全场都只跑 1 轮"，
        # 结果连 mock/nollm 的 best-of-K 一起被砍掉——它们免费，
        # 砍掉纯属白丢精度。real 那一档小，是因为它每条十几秒**还花钱**。
        tier_repeat = args.repeat if args.repeat is not None else TIER_DEFAULT_REPEAT[tier]
        print(f"--- {tier} 档（每配置 {n} 个请求 × {tier_repeat} 轮，"
              f"预热 {args.warmup}）---")
        # 零件在这一档开跑之前造。见 `_heavy_parts` 的注释：
        # 模型加载是 20 秒满载，会把机器推进慢档，所以它必须夹在
        # "上一档测完"和"这一档开测"之间，不能一上来全造完。
        t0 = time.perf_counter()
        parts[tier] = _heavy_parts(tier)
        print(f"  装配耗时 {time.perf_counter() - t0:.1f}s"
              f"（含模型加载；之后机器会进慢档，见报告开头）")

        skeleton = [(conc, scenario) for conc in concs_for(tier, args)
                    for scenario in ("fanout", "same")]
        for rep in range(tier_repeat):
            # **第 2 轮把顺序整个倒过来。** 机器是会被跑慢的，"谁先跑谁占便宜"，
            # 顺序固定的话这个便宜永远落在同一批配置头上，横向比较就废了。
            order = skeleton if rep % 2 == 0 else list(reversed(skeleton))
            for conc, scenario in order:
                pre_ms = await yardstick()
                yards.append(pre_ms)
                app = make_app(tier, parts[tier])  # 全新会话表，见文件头
                r = await run_config(app, tier, scenario, conc, n, args.warmup)
                r.yard_ms = pre_ms
                r.repeat = tier_repeat
                cur = best.get(r.key)
                if cur is None or r.qps > cur.qps:
                    best[r.key] = r
                    print(f"  {r.row(min(yards))}  ← 目前最快")
                else:
                    print(f"  {r.row(min(yards))}")
        print()

    # 按"档 → 并发 → 场景"的固定顺序输出，别让输出顺序随运行顺序变。
    results: list[Result] = []
    for tier in args.tier:
        for conc in concs_for(tier, args):
            for scenario in ("fanout", "same"):
                results.append(best[(tier, scenario, conc)])
    ref_ms = min(yards) if yards else float("nan")

    # ---- 报告 ---------------------------------------------------------- #
    lines = [
        "# F32 压测报告",
        "",
        "> **这张表怎么读**：三档之间的差是**检索**和**模型**贡献的，",
        "> 不是服务层的差异——HTTP 层、会话层、事件层、图，三档完全相同。",
        "",
        "## ⚠️ 先说机器：这份表差点就是错的",
        "",
        "**第一版压测得出了完全相反的结论**，过程值得完整写一遍，",
        "因为它是这类工作里最容易犯的错：**把机器的抖动当成系统的性质。**",
        "",
        "1. 同一个配置（mock / fanout / 并发 32 / 96 个请求），单独跑时 98 QPS，",
        "   放进整跑里只有 54 QPS。差一倍。",
        "2. 顺着「越跑越慢」查：先怀疑会话表越堆越大——查 `SessionStore`，",
        "   `OrderedDict` + LRU，取会话 O(1)，**不是它**；",
        "   把会话表清空再 `gc`，QPS 一点没回来（55.5 → 57.0），**不是内存**。",
        "3. 拿一段**与项目毫无关系的纯 Python 计算**（16 线程各烧定长循环）",
        "   连跑 15 秒：头 1.6 秒每轮约 72ms，第 5 秒变成约 122ms，",
        "   之后稳定在 1.4~1.6 倍。**这台机器（云主机，有 CPU 配额）",
        "   持续满载几秒后就进「慢档」。**",
        "4. 当时的「决定性一步」：**把并发 32 提到最前面跑**（此刻机器还快），",
        "   得 98.5 QPS，和并发 1 的 89.2 差不多。于是结论写成了",
        "   「并发不掉吞吐，掉的是机器状态」。",
        "",
        "**第 4 步那个结论后来被推翻了，而且推翻它的正是上面这套防坑机制。**",
        "",
        "把测量方式修好之后（每配置 6 轮取最快、偶数轮把配置顺序整个倒过来），",
        "只跑并发 1 和并发 32 这两个点重新量：`mock` / `fanout` 从并发 1 到 32，",
        "QPS 掉到**一半多一点**（具体数见下面那张表，两行都在；",
        "那两条最快的标尺分别是 1.00x 和 1.26x，说明**两次都跑在快机器上**，",
        "不是机器慢造成的）。顺序已经倒过来跑过了，所以这个差也不可能是顺序偏差。",
        "",
        "那之前那个 98.5 是什么？**单次采样。** 重复六次就再也测不出来。",
        "一句话：**一个样本打不过六个样本**，哪怕那一个样本当时看着像「决定性证据」。",
        "（顺带一个教训：越是「一步定案」的实验越要多想一层——",
        "它把结论带向的方向，恰好是作者本来希望的方向。）",
        "",
        "### 那到底为什么掉？（两个实验，不是一个）",
        "",
        "**实验一：对照组——多线程跑纯 Python 亏不亏。**",
        "同一份总计算量，①一个线程串行做完，②丢给 20 个线程同时做。",
        "总工作量一样，理想情况下比值应该是 1.00。实测（`--probe`）：",
        "",
        "| 计算形状 | 比值 ②/① |",
        "|---|---:|",
        "| 紧凑整数循环（只碰寄存器） | **1.00x** |",
        "| 大量小函数调用 | 0.96x |",
        "| **拼命分配内存（拼字符串、建字典）** | **1.57x** |",
        "",
        "**形状不同，结论完全不同**——所以「多线程跑 Python 亏不亏」这个问题",
        "本身是错的，得看那段代码在不在堆上反复申请释放。",
        "而**真实业务代码就是最后那一种**（拼字符串、建字典、建对象、序列化 JSON），",
        "亏 1.57x，和 app 里掉的那 1.8x 是同一个量级。",
        "机制说白了：GIL 同一时刻只让一个线程跑 Python，被换下来的线程要把手上的",
        "东西放下、换下一个上来；线程只要在不停地申请/释放内存，这个「换手」就特别贵。",
        "",
        "**实验二：直接把 app 的线程池压到 1 个线程（`--threads 1`）。**",
        "上面证明的是「这台机器的性格」，这一个证明的是「就在这个服务上」。",
        "把 `asyncio.to_thread` 的池子从 20 压到 1，同一个 `mock` / `fanout` 重跑：",
        "",
        "| `mock` / `fanout` | 线程池 20（默认） | 线程池 1 |",
        "|---|---:|---:|",
        "| 并发 1 | 101.6 | 99.6 |",
        "| 并发 32 | **57.1** | **90.2** |",
        "",
        "（这一小表是单独跑的诊断，命令：`--tier mock --conc 1 --conc 32 --tier-n 96 --repeat 6`，",
        "加不加 `--threads 1` 各跑一次。数字会变，结论不该变。）",
        "",
        "**池子从 20 压到 1，并发 32 的吞吐从 57 涨到 90，p50 从 539ms 降到 337ms。**",
        "线程少了反而快——因为活是纯 Python、本来就被 GIL 串着，",
        "多出来的线程只是在抢锁和上下文切换上白花功夫。",
        "",
        "**所以修法只有一个方向：加进程**（`uvicorn --workers N`），",
        "不是在这个进程里加并发。本项目没做多进程部署，所以这个数不在这张表里。",
        "",
        "**顺带解释了 `same` 为什么反而好看。** 表里 `same` 在并发 32 上比 `fanout` 高。",
        "原因不是「锁好」，而是**那把会话锁不小心把并发压回了 1**",
        "（同一个会话的请求本来就该排队），于是正好绕开了上面这个线程争抢。",
        "**它是被排队救的，不是被锁救的。**",
        "这一条值得单独记：一把为了「正确性」（防止两个请求互相盖写）加的锁，",
        "顺带带来了一点点性能好处，但那是副作用，不是设计目标——",
        "**不能因为「看着更快」就以为排队是好事。**",
        "",
        "### 对付机器抖动的办法（以及一个被否掉的办法）",
        "",
        "**否掉的**：每个配置前量一把标尺、按倍率折算 QPS。不行的原因是——",
        "标尺一次只有 70~130ms，噪声约 ±10%，而倍率要**乘到 QPS 上**，",
        "噪声跟着放大；实测折算完出现过 137 QPS 这种**比冷机还高**的怪数。",
        "**用一把不准的尺子去量，只会得到更不准的数。**",
        "",
        "**在用的**：每个配置跑 **3 轮**，取最快的那轮（`real` 档只跑 1 轮，",
        "因为它每条十几秒**还花钱**）。它成立只需要一个前提——",
        "**机器只会比它自己的能力强项慢，不会更快**（被邻居抢占、被配额压住，",
        "都只让数字变小）。所以「最快的一遍」是这台机器能力的下界估计，",
        "而**这个前提不需要标尺准**。另外第 2 轮的配置顺序整个倒过来跑，",
        "「谁先跑谁占便宜」就不会固定落在某个配置头上。",
        "",
        "标尺没丢，它降级成了一个**诊断列**（`标尺`）：量这个配置时机器",
        f"比全程最快时慢了几倍。本轮全程最快的标尺是 **{ref_ms:.0f}ms**。",
        "数不好看的时候先看这一列，别急着改代码。",
        "",
        "⚠️ **但要说清 best-of-K 救不了什么：** `nollm` 那一档的模型加载本身就是",
        "20 秒满载，会把机器压进慢档，于是那一档的 3 轮全都慢，取最快也还是慢。",
        "所以本表**绝对值只在同一档内可比**；跨档比大小（比如拿 mock 的 QPS",
        "去比 nollm 的）本来也没意义——测的就不是同一件事。",
        "标尺那一列就是留给读者判断「这条数是快机器上量的还是慢机器上量的」。",
        "",
        "## 测量条件",
        "",
        "- 请求直接喂 ASGI app（`httpx.ASGITransport`），**没有经过 uvicorn 的网络栈**，",
        "  所以 `mock` 档是服务层吞吐的**上界**。",
        f"- 单进程单事件循环；图跑在 `asyncio.to_thread` 默认线程池里，"
        f"上限 `min(32, cpu+4)` = **{tp}**。",
        f"- CPU 核心 {os.cpu_count()}；每个配置前 {args.warmup} 个预热请求不计入统计。",
        "- 每个配置用一个**全新的 app**（干净会话表），配置之间不互相污染。",
        "- `real` 档**真调 API、真花钱**，所以每配置只有 4 个计时请求（外加 2 个预热），",
        "  一共 24 次调用。**样本这么少，p95 基本没有统计意义**，当量级看。",
        "- 问题集是 8 条混合题（法条问答 / 计算题 / 超纲 / 闲聊），",
        "  **不是单一题型**——单一题型的 QPS 只代表那个题型。",
        "- 每配置 3 轮（`real` 档 1 轮），表里是**最快的那轮**；",
        "  p50/p95/p99 也取自那一轮（`轮数` 那一列就是各档实际跑了几轮）。",
        "- ⚠️ **整场运行期间机器大多处在慢档**（标尺那一列普遍 1.2~1.5x），",
        "  所以这些绝对数字**只当量级看**。换台安静的机器重跑，绝对数会明显更好，",
        "  但「加并发掉吞吐」「`same` 是一条直线」这两条结论都不该变——",
        "  它们靠的是同一台机器上的相对比较，不是绝对值。",
        "",
        "| 档 | 场景 | 并发 | 轮数 | 请求数 | p50 (ms) | p95 (ms) | p99 (ms) | QPS | 标尺 | 失败 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [r.row(ref_ms) for r in results]
    lines += [
        "",
        "## 怎么读这张表",
        "",
        "1. **`mock` 档是服务层吞吐的上界**，不是「系统能扛这么多请求」。",
        "   它测的是 HTTP + 事件序列化 + 会话锁，模型和检索都是假的。",
        "2. **加并发不但不提吞吐，还把吞吐压掉一半以上。**",
        "   表里 `mock` / `fanout` 从并发 1 到 32，QPS 一路往下走（98.3 → 55.8），",
        "   而 p50 从十几毫秒涨到几百毫秒——**延迟涨、吞吐还降**，",
        "   两头都变差，这不是排队，是**干活本身变慢了**。",
        "   ⚠️ **但光看这张表证明不了这一条，得说清楚：**",
        "   并发 1 那一行的标尺是 1.00x（机器正快），并发 32 那几行是 1.6x",
        "   （机器在慢档）。这台机器慢档和快档本来就差 1.4~1.6 倍，",
        "   而 QPS 的落差是 1.76 倍——**两个数太接近，光靠本表分不清",
        "   哪部分是并发造成的、哪部分是机器慢**。",
        "   真正钉死这一条的是**另一次等标尺的重跑**：只跑并发 1 和 32 两个点、",
        "   每点 6 轮取最快、偶数轮倒序，两行都落在快机器上（标尺 1.00x / 1.26x），",
        "   结果是 101.6 → 57.1。再加下面那两个实验定机制。",
        "   **所以正确的说法是「两次独立测量都指向同一个方向」，不是「这张表自己就能看出来」。**",
        "   原因不是「服务写得不好」，是**线程在抢 GIL**：",
        "   上面那两个实验一个证明了这台机器上「分配密集的纯 Python 活，",
        "   20 个线程干就是比 1 个线程慢 1.57 倍」，另一个直接把 app 的池子",
        "   压到 1 个线程、看着吞吐从 57 回到 90。",
        "   **所以这一条的正确读法是：在本进程里加并发是负收益。**",
        "   真要提吞吐，正确做法是**加进程**（`uvicorn --workers N`），",
        "   而不是在这个进程里加并发——本项目没做多进程部署，所以不在这张表里。",
        "   ⚠️ 也**别拿这一档的数去说「服务只能扛 57 QPS」**：",
        "   这一档的活是假的、零 IO，现实里每个请求大部分时间在等模型，",
        "   线程大部分时间闲着，抢 GIL 的情况比这里轻得多。",
        "3. **`nollm` 档的 QPS 掉到个位数**，那才是检索链路的真实成本：",
        "   编码 + RRF 融合 + 精排（rerank 是大头）。",
        "   对照 `reports/ablation.md` 里精排把 p50 从 20ms 拉到 221ms——",
        "   这里的 p50 与之吻合。**检索才是这个系统的耗时主体，不是 HTTP 壳。**",
        "   顺带：这一档加并发**没有**掉吞吐，反而略微上升——",
        "   因为活是计算密集但耗时长，线程在等的时候多，抢锁的相对代价就小了。",
        "   （这正好是第 2 条那个「假活夸大争抢」的旁证。）",
        "4. **`real` 档是唯一能对外说的那个数——端到端，含模型网络往返。**",
        "   它一条要**十几到几十秒**，所以 QPS 是 0.0x 这种量级",
        "   （表里低于 1 的 QPS 保留两位小数，不然会显示成 0.0，看着像服务死了）。",
        "   ⚠️ **这一档的 p95 只有个位数样本，基本没有统计意义**，",
        "   它的用途是给出「一轮问答到底要等多久」的量级，不是拿来做 SLO。",
        "   真要做容量规划，得按「并发用户在等」来算：一条十几秒、",
        "   单进程同时只伺候得动个位数，再往上就得靠加进程。",
        "5. **`same` 那一列几乎是条直线，而且在并发高的时候它比 `fanout` 好看。**",
        "   原因是**会话锁把并发压回了 1**——同一个会话的请求本来就该排队，",
        "   于是顺手绕开了第 2 条那个线程争抢。**它是被排队救的，不是被锁救的。**",
        "   这一条值得单独讲，因为「加锁 = 变慢」是很容易脱口而出的直觉：",
        "   这里恰恰相反，但那是副作用，**不能反过来推出「排队是好事」**——",
        "   锁的正当理由是正确性（两个请求同时写会互相盖掉），不是性能。",
        "   顺带一个方法论提醒：`same` 在**小请求数**下会测出很低的 QPS",
        "   （几十个请求里掺一个几百毫秒的冷启动请求，它就能把整段墙钟时间撑大），",
        "   这不是服务性质，是样本太少。**请求数不能太少，理由和 best-of-K 一样。**",
    ]
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已写入 {out}")

    print("\n| 档 | 场景 | 并发 | p50 | p95 | QPS | 标尺 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        rel = f"{r.yard_ms / ref_ms:.2f}x" if ref_ms == ref_ms and ref_ms > 0 else "—"
        print(f"| {r.tier} | {r.scenario} | {r.conc} | {r.q(0.5) * 1000:.0f}ms | "
              f"{r.q(0.95) * 1000:.0f}ms | {_fmt_qps(r.qps)} | {rel} |")
    total_err = sum(r.errors for r in results)
    if total_err:
        print(f"\n⚠️ 有 {total_err} 个请求失败——**先查失败，再看延迟**。")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F32 压测：并发与延迟（mock/nollm/real 分开报）")
    ap.add_argument("--tier", action="append", choices=["mock", "nollm", "real"],
                    default=None, help="可重复；默认只跑 mock（免费）")
    ap.add_argument("--tier-n", type=int, default=None,
                    help="每个配置的请求数（不传用各档默认：mock 96 / nollm 32 / real 6）")
    ap.add_argument("--warmup", type=int, default=2, help="每配置的预热请求数，不计入统计")
    ap.add_argument("--repeat", type=int, default=None,
                    help="每个配置跑几轮取最快（不传：mock/nollm 3 轮，real 1 轮）。"
                         "**别为了省时间调成 1**：这台机器会被跑慢，"
                         "只跑一轮的话「谁先跑谁占便宜」会直接变成结论。")
    ap.add_argument("--conc", action="append", type=int, default=None,
                    help="只跑指定的并发（可重复，例如 `--conc 1 --conc 32`）。"
                         "取值必须是这一档默认表里的那些（mock 1/4/16/32、nollm 1/4/8、"
                         "real 1/2）——随手补一个没测过的并发，两份报告的行就对不上了。")
    ap.add_argument("--probe", action="store_true",
                    help="只跑「多线程跑纯 Python 亏不亏」那个对照组然后退出，不压测、不写报告。"
                         "并发一高吞吐就掉的时候，先跑它分清「是服务的问题」还是「机器 + GIL 的问题」。")
    ap.add_argument("--threads", type=int, default=None,
                    help="**诊断用**：把 app 干活的线程池上限改成这个数"
                         "（默认不传 = `asyncio.to_thread` 的默认值 min(32, cpu+4)）。"
                         "传 `--threads 1` 再跑一遍并发 32，就能分清「并发掉吞吐」"
                         "到底是线程互相抢锁造成的，还是服务代码自己的问题。"
                         "**这个开关只影响被测的 app，不影响标尺**（标尺自带一个池子）。")
    ap.add_argument("--out", default="reports/qps.md")
    ap.add_argument("--log-level", default="WARNING",
                    help="压测时的日志级别（默认 WARNING：每个请求都会打一条「一轮结束」，"
                         "几百行日志会把结果淹没）。想看细节传 INFO。")
    args = ap.parse_args(argv)
    args.tier = args.tier or ["mock"]

    # 把日志压掉，理由见 `--log-level`。两处都要设：
    # ① 我们自己的日志走 structlog（`configure_logging`）；
    # ② httpx / httpcore 的 "HTTP Request: POST ... 200 OK" 走标准库，
    #    它们不归 structlog 管，得单独把那个 logger 的级别调高。
    from juris_agent.observability import configure_logging

    configure_logging(level=args.log_level)
    import logging

    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.probe:
        # 对照组很轻（几秒），但它自己也会把机器跑热，所以**必须单独跑**，
        # 不能塞在压测中间——塞进去那次满载会把后面几个配置一起拖慢。
        pr = asyncio.run(thread_probe())
        print_thread_probe(pr, min(32, (os.cpu_count() or 1) + 4))
        return 0

    if "real" in args.tier:
        print("⚠️ `real` 档会**真的调用 API 并计费**。"
              f"每配置 {(args.tier_n or TIER_DEFAULT_N['real'])} 个请求 × "
              f"{len(CONCURRENCIES['real'])} 个并发 × 2 个场景。")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
