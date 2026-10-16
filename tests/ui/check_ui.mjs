/* 拿**真后端**的 schema 和 payload 喂给前端的表单生成器和结果渲染器。
 *
 * 这一步回答了三个"只看代码看不出"的问题：
 *   ① 7 个工具，每一个的 schema 都能生成出表单吗（会不会有认不出的类型炸掉）？
 *   ② 枚举字段真的变成下拉框了吗、可空布尔真的变成三选一了吗？
 *   ③ 四种形状完全不同的结果（算式/字符串发现/对象发现/嵌套数组）都能渲染吗？
 *
 * 用法：先起服务（默认 8811），再 node tests/ui/check_ui.mjs
 */
import { api, realFetch } from "./dom_shim.mjs";

// DOM 桩把全局 fetch 换成了假货，这里换回真的。
const fetch = realFetch;

const BASE = process.env.BASE || "http://127.0.0.1:8811";
let failed = 0;
const ok = (cond, msg) => {
  console.log((cond ? "  ✅ " : "  ❌ ") + msg);
  if (!cond) failed++;
};

/** 把一棵 DOM 桩摊平成文本，用来断言"某个字段真的被画出来了"。 */
function flat(node, depth = 0) {
  let s = node._html + node._text + (node.title || "");
  for (const c of node.children) s += " " + flat(c, depth + 1);
  return s;
}
const find = (node, pred) => {
  if (pred(node)) return node;
  for (const c of node.children) { const r = find(c, pred); if (r) return r; }
  return null;
};
const all = (node, pred, out = []) => {
  if (pred(node)) out.push(node);
  for (const c of node.children) all(c, pred, out);
  return out;
};

const tools = (await (await fetch(`${BASE}/api/tools`)).json()).tools;
console.log(`\n拿到 ${tools.length} 个工具\n`);

// ---- ① 每个工具都能生成表单 ----
console.log("① schema → 表单");
for (const t of tools) {
  const props = t.args_schema.properties || {};
  let built;
  try {
    built = api.schemaToForm(t.args_schema);
  } catch (e) {
    ok(false, `${t.name}：生成表单时抛异常 —— ${e.message}`);
    continue;
  }
  const nProps = Object.keys(props).length;
  ok(built.fields.length === nProps,
     `${t.name}：${nProps} 个字段全部生成出输入框`);
  // 每个字段都得有输入控件，不能有"生成了个空壳"
  const noInput = built.fields.filter(w => !w._input);
  ok(noInput.length === 0, `${t.name}：没有字段是空壳${noInput.length ? "（" + noInput.length + " 个）" : ""}`);
  // 必填标记的数量应当等于 required 的数量
  const marked = built.fields.filter(w => w._required).length;
  ok(marked === (t.args_schema.required || []).length,
     `${t.name}：必填标记 ${marked} 个 = schema 的 required ${(t.args_schema.required || []).length} 个`);
}

// ---- ② 关键字段的控件类型对不对 ----
console.log("\n② 控件类型");
const sev = tools.find(t => t.name === "calc_severance");
const sf = api.schemaToForm(sev.args_schema);
const byName = (n) => sf.fields.find(w => w._name === n);

const cause = byName("cause");
ok(cause._input.tagName === "SELECT", "cause（枚举）→ 下拉框");
ok(cause._input.children.length === 11,
   `cause 下拉框有 11 项（1 个"请选择" + 10 个枚举值），实得 ${cause._input.children.length}`);
const optTexts = cause._input.children.slice(1).map(o => o.text);
ok(optTexts.every(x => x && x !== "undefined"),
   "cause 的每个选项都带上了人话解释（没有裸的英文枚举名）");
ok(cause._input.children[1].title.includes("employee_forced"),
   "选项的 title 里保留了原始枚举值（悬停能对回代码）");

const notice = byName("notice_given");
ok(notice._input.tagName === "SELECT" && notice._input.children.length === 3,
   "notice_given（可空布尔）→ 三选一，不是勾选框");

const months = byName("months_of_service");
ok(months._input.tagName === "INPUT" && months._input.type === "number",
   "months_of_service（整数）→ 数字输入框");
ok(months._input.min === 0, "months_of_service 带上了 schema 里的 minimum=0");

const lim = tools.find(t => t.name === "calc_limitation");
const lf = api.schemaToForm(lim.args_schema);
const interrupted = lf.fields.find(w => w._name === "interruptions");
ok(interrupted._input.tagName === "INPUT" && interrupted._input.type === "text",
   "interruptions（日期数组）→ 逗号分隔的文本框");
ok(interrupted._input.placeholder.includes("逗号"), "数组字段的 placeholder 说明了填法");

const know = lf.fields.find(w => w._name === "know_date");
ok(know._input.tagName === "INPUT" && know._input.type === "date",
   "know_date（format=date）→ 日期选择器");

// ---- ③ 读表单：填进去的能原样读出来 ----
console.log("\n③ 表单 → 参数");
const sf2 = api.schemaToForm(sev.args_schema);
const f2 = (n) => sf2.fields.find(w => w._name === n);
f2("cause")._input.value = "no_fault_dismissal";
f2("months_of_service")._input.value = "26";
f2("avg_monthly_wage")._input.value = "12000";
f2("notice_given")._input.value = "false";
f2("last_month_wage")._input.value = "12000";
let r = api.readForm(sf2.fields);
ok(r.missing.length === 0, "必填项都填了 → 没有报缺");
ok(r.bad.length === 0, "没有格式错误");
ok(r.args.cause === "no_fault_dismissal", "枚举读回字符串");
ok(r.args.months_of_service === 26, "整数读回 number（不是字符串）");
ok(r.args.notice_given === false, "三选一的「否」读回布尔 false");
ok(!("local_avg_monthly_wage" in r.args), "留空的选填项不出现在参数里（不是 null）");

// 必填没填 → 要报出来，而且**不发请求**
const sf3 = api.schemaToForm(sev.args_schema);
const r3 = api.readForm(sf3.fields);
ok(r3.missing.length === 3, `空表单报出 3 个必填（实得 ${r3.missing.length}）`);

// 数字框里塞非数字 → 报格式错，而不是悄悄变成 NaN 发出去
const sf4 = api.schemaToForm(sev.args_schema);
sf4.fields.find(w => w._name === "months_of_service")._input.value = "两年";
const r4 = api.readForm(sf4.fields);
ok(r4.bad.includes("工作月数"), "数字框填了中文 → 报格式错（不会把 NaN 发出去）");

const lf2 = api.schemaToForm(lim.args_schema);
lf2.fields.find(w => w._name === "interruptions")._input.value = "2025-01-01, 2025-06-01";
const r5 = api.readForm(lf2.fields);
ok(Array.isArray(r5.args.interruptions) && r5.args.interruptions.length === 2,
   "逗号分隔的数组读回真数组（2 项）");

// ---- ④ 结果渲染：四种形状都能吃 ----
console.log("\n④ payload → 结果面板");
const CALLS = [
  ["calc_severance", { cause: "no_fault_dismissal", months_of_service: 26, avg_monthly_wage: 12000,
                       notice_given: false, last_month_wage: 12000 }],
  ["calc_overtime", { kind: "statutory_holiday", monthly_wage: 8000, days: 3 }],
  ["check_probation", { term_months: 36, agreed_probation_months: 6 }],
  ["calc_limitation", { know_date: "2025-03-01", today: "2026-10-16", interruptions: ["2025-09-01"] }],
  // ⚠️ ContractInput 是**结构化字段**，不是自由文本条款。
  // 一开始我按"丢一段合同文字进去"填，结果 0 条发现——那不是渲染器的 bug，
  // 是**我喂给它一个没有任何可查项的输入**（所有字段都是可选的，
  // 不提供就不检查）。这组值能同时触发"没签书面合同"和"试用期超期"。
  ["review_contract", { written_contract: false, months_without_written_contract: 3,
                        monthly_wage: 8000, term_months: 12, probation_months: 6 }],
  ["search_law", { query: "试用期 最长 期限" }],
];
for (const [name, args] of CALLS) {
  const res = await fetch(`${BASE}/api/tools/${name}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  const d = await res.json();
  if (d.status !== "ok") { ok(false, `${name}：后端返回 ${d.status}，没法测渲染`); continue; }
  let node;
  try {
    node = api.renderPayload(d.payload);
  } catch (e) {
    ok(false, `${name}：渲染时抛异常 —— ${e.message}`);
    continue;
  }
  const text = flat(node);
  ok(text.length > 20, `${name}：渲染出内容（${text.length} 字符）`);

  if (name === "calc_severance") {
    ok(text.includes("42000"), "  算式区画出了合计金额 42000");
    ok(all(node, n => n.tagName === "LI").length >= 5, "  算式渲染成了有序列表");
  }
  if (name === "calc_limitation") {
    ok(text.includes("时效中断链"), "  中断链单独画出来了（不是只塞进原始 JSON）");
  }
  if (name === "review_contract") {
    const fx = all(node, n => n.classList.contains("finding"));
    ok(fx.length > 0, `  体检发现渲染成 ${fx.length} 张卡片`);
    ok(fx.some(n => n.classList.contains("illegal")), "  违法条款带上了 illegal 分级");
  }
  if (name === "check_probation") {
    const fx = all(node, n => n.classList.contains("finding"));
    ok(fx.length > 0, `  findings 是纯字符串也渲染成了 ${fx.length} 张卡片`);
  }
  if (name === "search_law") {
    ok(all(node, n => n.classList.contains("cite")).length > 0, "  命中的法条渲染成卡片");
  }
  // 任何工具的结果都该有"原始值"折叠块，方便深挖
  ok(all(node, n => n.classList.contains("raw")).length === 1, "  有「全部字段（原始值）」折叠块");
}

// ---- ⑤ XSS：工具返回的文本里带标签要被转义 ----
console.log("\n⑤ 转义");
const evil = api.renderPayload({ steps: ["<img src=x onerror=alert(1)>"], warnings: ["<b>x</b>"] });
ok(!flat(evil).includes("<img"), "算式里的 HTML 标签被转义了");
ok(!flat(evil).includes("<b>x</b>"), "提醒里的 HTML 标签被转义了");
ok(api.inlineMd("看**这里**").includes("<strong>这里</strong>"), "**加粗** 生效");
ok(api.inlineMd("行内 <script> 被转义").includes("&lt;script&gt;"), "先转义再套标签（顺序对）");

// 转义必须**在套标签之前**做完。只测"尖括号被转义"是不够的——
// 顺序写反（先拼 <strong> 再整体转义）会让标签本身也被转义掉，
// 页面上显示字面的 <strong>。所以这里同时断言"标签在、源码不在"。
const order = api.inlineMd("**<b>粗</b>**");
ok(order.includes("<strong>") && !order.includes("&lt;strong&gt;"), "标签没被二次转义");
ok(order.includes("&lt;b&gt;"), "模型写的标签仍在加粗里被转义");

// 这三个是 Markdown 渲染器**新引入**的攻击面（原来只有 esc 一条路）：
// 链接、图片、以及代码块里的内容，都得照样转义。
classic: {
  const cases = [
    ["代码块", api.inlineMd("`<script>alert(1)</script>`")],
    ["标题", api.mdToHtml("## <img src=x onerror=alert(1)>")],
    ["表格", api.mdToHtml("| a | b |\n|---|---|\n| <img src=x> | y |")],
    ["列表", api.mdToHtml("- <img src=x onerror=alert(1)>")],
    ["引用块", api.mdToHtml("> <img src=x onerror=alert(1)>")],
  ];
  for (const [what, html] of cases) {
    ok(!html.includes("<img") && !html.includes("<script"),
       `${what}里的标签也被转义`);
  }
}

// ---- ⑥ Markdown 渲染 ----
// 支持范围是**数出来的**（见 web/index.html 文件头），所以这里断言的就是
// 那几样真的出现过的东西，外加几个"写错了也不能把正文吃掉"的边界。
console.log("\n⑥ Markdown");

ok(api.mdToHtml("## 依据").includes("<h3>依据</h3>"), "## → h3（不是 h2：页头在用）");
ok(api.mdToHtml("# 标题").includes("<h3>"), "# → h3（跟 ## 同级）");
ok(api.mdToHtml("### 细项").includes("<h4>细项</h4>"), "### → h4");
ok(api.mdToHtml("#### 更细").includes("<h5>"), "#### → h5（到顶了，不再降）");
ok(!api.mdToHtml("## 依据").includes("<h1>"), "绝不生成 h1/h2（会和页面自己抢）");

ok(api.mdToHtml("- 甲\n- 乙").includes("<ul><li>甲</li><li>乙</li></ul>"), "无序列表");
ok(api.mdToHtml("1. 甲\n2. 乙").includes("<ol><li>甲</li><li>乙</li></ol>"), "有序列表");
ok(api.mdToHtml("- 甲\n  续行").includes("<li>甲<br>续行</li>"),
   "列表续行接回上一项（不另起一段）");
const loop = api.mdToHtml("- 甲\n\n- 乙");
ok(loop.includes("<ul>") && loop.split("<ul>").length === 2, "列表中间的空行不断成两个 ul");

// ⚠️ 分割线必须在列表**之前**判：`---` 也满足 `- ` 那条列表正则，
// 判反了它就是一个内容为 `--` 的列表项。这是这套解析器最脆的一处。
const hr = api.mdToHtml("前\n\n---\n\n后");
ok(hr.includes("<hr>"), "--- 渲染成分割线");
ok(!hr.includes("<li>"), "--- 没有变成 `-` 列表项（顺序对）");
// ⚠️ 紧贴在上一行后面的 `---`（中间没空行）也要断掉段落。
// 真实语料里 `---` 前后恰好都有空行，所以漏渲染统计是 0——
// 但那是**数据碰巧**，不是代码保证的。这条断言管的是代码这一侧。
ok(api.mdToHtml("前一行\n---\n后一行").includes("<hr>"),
   "没有空行隔开的 --- 也断段（不靠数据碰巧）");
ok(api.mdToHtml("***").includes("<hr>") && api.mdToHtml("___").includes("<hr>"),
   "* * * 和 ___ 也算分割线");

const tbl = api.mdToHtml("| 情形 | 倍数 |\n|---|---|\n| 违法解除 | 2 |");
ok(tbl.includes("<table>") && tbl.includes("<th>情形</th>"), "管道表：表头成 th");
ok(tbl.includes("<td>违法解除</td>"), "管道表：数据行成 td");
// 宽表格要能自己横向滚，而不是把整个页面撑宽。
ok(tbl.includes('class="tw"') && tbl.indexOf('class="tw"') < tbl.indexOf("<table>"),
   "表格套了 .tw 横向滚动容器（不在调用处，在渲染器里）");
// 判据是"下一行是分隔行"。只看第一行的话，正文里一个孤零零的 `|` 会被误判成表格。
ok(!api.mdToHtml("| 这不是表格，下面没有分隔行").includes("<table>"),
   "只有一行 | 不认成表格（要下一行是分隔行才算）");

ok(api.mdToHtml("> 引用一句").includes("<blockquote>"), "引用块");
ok(api.mdToHtml("> 外层\n> > 内层").includes("<blockquote>") &&
   api.mdToHtml("> 外层\n> > 内层").includes("<blockquote><p>内层</p></blockquote>"),
   "引用块嵌套（渲染器是递归的）");

ok(api.mdToHtml("第一行\n第二行").includes("第一行<br>第二行"), "段落里的单换行 → <br>");
ok(api.mdToHtml("甲\n\n乙").includes("<p>甲</p><p>乙</p>"), "空行分成两段");

// ⚠️ **故意不支持 `*斜体*`**：一次都没在真实答案里出现过，
// 而它的正则会吃掉算式里的乘号。这条断言就是钉死这个决定——
// 哪天有人"顺手补上斜体"，这里会立刻红。
const math = api.mdToHtml("12000 * 3.5 * 2");
ok(!math.includes("<em>") && math.includes("12000 * 3.5 * 2"), "乘号没被当成斜体标记");

// `**` 不跨行：跨行的多半是模型写歪了，宁可原样显示，
// 也别把半篇答案吞进一个 <strong> 里。
ok(api.inlineMd("**没闭合\n下一行").includes("**没闭合"), "跨行的 ** 不当加粗（不吞正文）");

// 行内代码里的 `*` 不能被加粗正则吃掉（占位符那一步存在的理由）
ok(api.inlineMd("`a * b`").includes("<code>a * b</code>"), "行内代码原样保留");

// ---- 引用标记：后端写 `[1]`，这里必须认，两种括法都认 ----
console.log("\n⑦ 引用标记");
const c1 = api.inlineMd("依据《劳动合同法》[1] 第 47 条", [1]);
ok(c1.includes('class="cite-ref" data-idx="1"'), "[1] 变成可点标记（后端就是这么写的）");
ok(c1.includes("[1]</span>"), "标记文字保持 [1] 原样，没换成 【1】");
ok(api.inlineMd("见【1】", [1]).includes('data-idx="1"'), "【1】也认（两种括法都认）");

const c2 = api.inlineMd("见 [1,2]", [1, 2]);
ok(c2.split('class="cite-ref"').length === 3 && c2.includes('data-idx="2"'),
   "[1,2] 拆成两个可点标记");
ok(api.inlineMd("见 [1][2]", [1, 2]).split('class="cite-ref"').length === 3, "[1][2] 也拆");

// 校验没过的不做链接。**整段原样留着**，不是只留坏的那个：
// 半截可点（[1] 点得动、[9] 点不动）比整段不可点更让人困惑。
const bad = api.inlineMd("见 [1] 和 [9]", [1]);
ok(bad.includes('data-idx="1"') && !bad.includes('data-idx="9"'), "没校验过的编号不链接");
ok(api.inlineMd("见 [9]", [1]).includes("[9]") &&
   !api.inlineMd("见 [9]", [1]).includes("cite-ref"), "都不合法时整段原样");

// 没有可校验的编号 → 一个都不链接。
// 别把"列表是空的"当成"全都算数"，那样点上去只会弹"没有对应的法条原文"。
ok(!api.inlineMd("见 [1]", []).includes("cite-ref"), "validate 列表为空时一律不链接");
ok(!api.inlineMd("见 [1]", undefined).includes("cite-ref"), "没传 validIndexes 时不链接");
// 编号 0 是假值。渲染这条路用 Number()/模板串，没问题；
// 出问题的是点击那条路（`ref.dataset.idx || 0` 会把 1 变成 0），
// 那条要真 DOM 事件才测得到，所以这里只钉住渲染这半边的行为。
ok(api.inlineMd("见 [0]", [0]).includes('data-idx="0"'), "编号 0 渲染成 data-idx=0，没被吞掉");

// ---- ⑧ 流式：写到一半的标记不能露出来 ----
console.log("\n⑧ 流式");
ok(!api.streamSafe("答案是 **加粗").includes("**"), "没配对的 ** 被藏起来");
ok(api.streamSafe("**甲** 和 **乙**").includes("**乙**"), "配好对的 ** 不动");
ok(api.streamSafe("**甲** 和 **乙").endsWith("和 "), "只砍最后那一个，前面不动");
ok(api.streamSafe("普通文本").includes("普通文本"), "没有 ** 时原样返回");

console.log(failed ? `\n❌ ${failed} 项没过` : "\n✅ 全部通过");
process.exit(failed ? 1 : 0);
