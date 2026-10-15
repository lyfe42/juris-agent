"""F34 门禁**判定逻辑**的测试。

⚠️ **这里测的不是"Recall 是多少"**，而是"门禁会不会骗人"。

一条门禁最危险的失败方式不是漏报，是**误报**：
它红了，你去查了半天，最后发现是基线和语料对不上——
这种事发生两三次，大家就学会忽略红色了，门禁就死了。

所以这里的重点全在**判定顺序**和**边界**上：
- 语料/评测集变了的时候，必须**拒绝比较**，而不是算出一个假涨跌；
- 容差之内的小抖动不该变红；
- 配置写错（指标名拼错）要和"真的退化"分开报，因为修法完全不同。

真正的端到端跑（载语料、建索引、算指标）在 `scripts/check_recall.py`
和 CI 的 `recall-gate` job 里，不在这里——这里全是纯函数，毫秒级。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_gate_module():
    """把 `scripts/check_recall.py` 当模块加载。

    `scripts/` 不是包（没有 `__init__.py`），因为它是给人 `python xxx.py` 跑的，
    不是给人 `import` 的。这里要 import 它的纯函数，所以按路径显式加载。
    """
    path = ROOT / "scripts" / "check_recall.py"
    spec = importlib.util.spec_from_file_location("check_recall_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # ⚠️ **这一行不能省，而且理由一点也不显然。**
    # `GateFile` 里有个字段是 `list[Gate]`。模块开了
    # `from __future__ import annotations`，所以那是一句**字符串注解**，
    # 由 pydantic 事后去解析。它解析时找的是 `sys.modules[类.__module__]` ——
    # 而 `spec_from_file_location` **默认不把模块登记进 `sys.modules`**。
    # 于是 pydantic 拿着模块名去查、查不到，报一句
    # `GateFile is not fully defined; you should define Gate, then call GateFile.model_rebuild()`。
    # 这句话把矛头指向了"类没定义好"，而实际上类定义得好好的，
    # **只是还没来得及登记**。手动登记一行就好。
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate_mod = _load_gate_module()
Gate = gate_mod.Gate
judge = gate_mod.judge


# --------------------------------------------------------------------------- #
# ① 判定顺序：拒绝比较必须压过数值比较
# --------------------------------------------------------------------------- #


class TestJudgmentOrder:
    """**顺序错了，整个门禁就是负资产。**"""

    def _gate(self, **kw) -> Gate:
        base = dict(
            name="g",
            min_values={"recall_art@5": 0.8},
            baseline={"recall_art@5": 0.9},
            baseline_hashes={"corpus_hash": "aaa", "gold_hash": "bbb", "split": "test"},
            tolerance=0.01,
        )
        base.update(kw)
        return Gate(**base)

    def test_stale_beats_fail(self):
        """语料变了 + 指标掉了 0.30 → 必须报 STALE，**不能**报 FAIL。

        这是这个文件里最重要的一条。两种判法的后果完全不同：
        - 报 FAIL：人去查一个不存在的回归，查一下午，最后发现是语料换了。
        - 报 STALE：人立刻知道"该重新标定了"。
        **数值掉得再多，只要数据换过，那个差就没有意义。**
        """
        v = judge(
            self._gate(),
            {"recall_art@5": 0.60},  # 掉了 0.30，远超容差
            {"corpus_hash": "CHANGED", "gold_hash": "bbb", "split": "test"},
        )
        assert v.status == "STALE", f"数据换了就不该比数值，实际判成了 {v.status}"
        assert "语料" in v.detail

    def test_gold_change_also_blocks_comparison(self):
        """评测集变了同样不能比。**很多人只防语料、忘了评测集。**"""
        v = judge(
            self._gate(),
            {"recall_art@5": 0.60},
            {"corpus_hash": "aaa", "gold_hash": "CHANGED", "split": "test"},
        )
        assert v.status == "STALE"
        assert "评测集" in v.detail

    def test_split_is_not_hashed(self):
        """`split` 不参与校验：它在配置里，改配置就会一起更新，哈希它只会自找麻烦。"""
        v = judge(
            self._gate(),
            {"recall_art@5": 0.90},
            {"corpus_hash": "aaa", "gold_hash": "bbb", "split": "dev"},  # 和基线不同
        )
        assert v.status == "PASS"

    def test_unknown_hashes_do_not_block(self):
        """基线里没记哈希（老文件 / 手工写的）→ 不拦，照常比。

        **这是刻意的宽松**：哈希齐全时拦得住，缺哈希时把它当成
        "这人知道自己在干什么"。反过来（缺哈希就报错）会让这个文件
        没法手工维护，而手工维护的能力比这点严格更值钱。
        """
        v = judge(
            self._gate(baseline_hashes={}),
            {"recall_art@5": 0.90},
            {"corpus_hash": "whatever", "gold_hash": "whatever", "split": "test"},
        )
        assert v.status == "PASS"

    def test_no_baseline_is_new_not_fail(self):
        """还没标定 → NEW。**不能判 PASS**（那叫"没有门禁"），也不能判 FAIL。"""
        v = judge(self._gate(baseline={}), {"recall_art@5": 0.90}, {})
        assert v.status == "NEW"


# --------------------------------------------------------------------------- #
# ② 容差与绝对下限
# --------------------------------------------------------------------------- #


class TestThresholds:
    def _gate(self, **kw) -> Gate:
        base = dict(
            name="g",
            min_values={"recall_art@5": 0.80},
            baseline={"recall_art@5": 0.90},
            baseline_hashes={},
            tolerance=0.01,
        )
        base.update(kw)
        return Gate(**base)

    def test_drop_beyond_tolerance_fails(self):
        v = judge(self._gate(), {"recall_art@5": 0.85}, {})
        assert v.status == "FAIL"
        assert "掉到" in v.detail

    def test_drop_within_tolerance_passes(self):
        """容差之内的小抖动不该变红。

        检索指标本身有噪声（jieba 词典热加载、浮点累加顺序、reranker 分数微差），
        卡到小数点后四位的话这条门禁每天都会红，然后就没人看了。
        **一条"经常红但都不是真事"的门禁，比没有门禁更坏。**
        """
        v = judge(self._gate(), {"recall_art@5": 0.895}, {})  # 跌 0.005 < 0.01
        assert v.status == "PASS"

    def test_exactly_at_tolerance_boundary_passes(self):
        """**边界是"跌得比容差少"，不是"少于等于"。** 跌到刚好 0.01 不算超。"""
        v = judge(self._gate(), {"recall_art@5": 0.89}, {})  # 跌 0.01，不 > 0.01
        assert v.status == "PASS"

    def test_absolute_floor_catches_a_value_that_never_moved(self):
        """**绝对下限防的是"一直很低但一直没变"这种情况。**

        基线就是 0.80、现在还是 0.80，差值比较完全满意；
        但如果下限是 0.85，这个值本来就不该被放行。
        只有相对基线的话，一次误标定能让一条烂门禁永远绿下去。
        """
        v = judge(
            self._gate(min_values={"recall_art@5": 0.85},
                       baseline={"recall_art@5": 0.80}),
            {"recall_art@5": 0.80},
            {},
        )
        assert v.status == "FAIL"
        assert "绝对下限" in v.detail

    def test_missing_metric_is_reported_as_a_config_error(self):
        """指标名拼错 → 报"配置错"，不是报"退化"。

        这两种的修法完全不同：一个是去看检索代码，一个是去改 YAML。
        混成一句话，人会先去查代码。
        """
        v = judge(self._gate(), {"recall_art@10": 0.9}, {})  # 少了 art@5
        assert v.status == "FAIL"
        assert "没算出来" in v.detail

    def test_the_first_failing_metric_decides_the_message(self):
        """多个指标同时退化时，报**第一个**（按 YAML 声明顺序）。

        顺序其实不重要，重要的是**只报一个**：一次报五条，
        读者会不知道该先修哪条，而它们多半是同一个原因。
        """
        g = self._gate(
            min_values={"a": 0.8, "b": 0.8},
            baseline={"a": 0.9, "b": 0.9},
        )
        v = judge(g, {"a": 0.5, "b": 0.5}, {})
        assert v.status == "FAIL"
        assert v.detail.startswith("a")


# --------------------------------------------------------------------------- #
# ③ 真实的门禁配置文件本身
# --------------------------------------------------------------------------- #


class TestGateConfigFile:
    """对**仓库里那份 `configs/recall_gate.yaml`** 的体检。

    这些断言不测算法，测的是"这份配置有没有在骗人"。它们会随着文件演化
    一直有用：**新加一条门禁忘了写 guards 或忘了标定，这里立刻红。**
    """

    @pytest.fixture
    def gates(self) -> list:
        import yaml

        raw = yaml.safe_load(
            (ROOT / "configs" / "recall_gate.yaml").read_text(encoding="utf-8")
        )
        return gate_mod.GateFile(**raw).gates

    def test_every_gate_declares_what_it_guards(self, gates):
        """每条门禁都要写清"我在守什么"。

        理由很实际：门禁红了的时候，看的人（包括三个月后的自己）
        得知道该去翻哪一块代码。**没有这句话，红色只是一个数字。**
        """
        for g in gates:
            assert g.guards.strip(), f"{g.name} 没写 guards——红了之后没人知道该查哪儿"

    def test_every_gate_is_calibrated(self, gates):
        """每条门禁都得有基线。

        一条没有基线的门禁永远返回 NEW，也就是**永远不红**——
        它在输出里占一行，让人以为这块被守住了。这正是要避免的那种"假防线"。
        """
        for g in gates:
            assert g.baseline, f"{g.name} 还没标定基线（跑 --update-baseline）"
            assert g.baseline_hashes, f"{g.name} 有基线但没有哈希，没法判断基线是否过期"

    def test_floors_are_not_above_the_baseline(self, gates):
        """下限高于基线 = 这条门禁**从标定那天起就没打算通过**。

        这种配置多半是手改数字时把两栏写串了。它不会立刻爆炸
        （因为基线那个测试会先过），但下一次跑就会红，而人会觉得莫名其妙。
        """
        for g in gates:
            for metric, floor in g.min_values.items():
                base = g.baseline.get(metric)
                if base is not None:
                    assert floor <= base + 1e-9, (
                        f"{g.name}：{metric} 的绝对下限 {floor} 比基线 {base} 还高，"
                        "这条门禁标定完就已经是红的了"
                    )

    def test_ci_runnable_gate_exists(self, gates):
        """至少要有一条 `requires: bm25` 的门禁。

        **CI 里跑不了 ml 组**（要下载 2.5GB torch）。如果所有门禁都要 ml，
        那 CI 里就是全 SKIP —— 门禁在 CI 里等于不存在，
        而它恰恰是唯一能拦住"改坏了"的地方。
        """
        assert any(g.requires == "bm25" for g in gates), (
            "没有任何一条门禁能在不装 ml extra 的环境跑，CI 里会全部 SKIP"
        )
