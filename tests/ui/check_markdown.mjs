/* 把 web/index.html 里的 Markdown 渲染器，拿**真实录制**的模型答案喂一遍。
 *
 * ## 为什么要单独有这一个文件，和 check_ui.mjs 的 ⑥⑦⑧ 组有什么区别
 *
 * `check_ui.mjs` 里那几组喂的是**我手写的字符串**。这有用，但它有个根本的盲区：
 * **我只能测到我想得到的写法。** 模型实际会写出什么，我猜不全——
 * 比如"表格"这一条，如果没有去数真实数据，我根本不会想到要做
 * （`|` 开头、下一行是 `|---|---|` 这种写法，光靠想象是想不出来的）。
 *
 * 所以这个文件喂的是 `fixtures/replay/llm/` 里录下来的真东西。
 * 两边的分工是：
 *   check_ui.mjs        手写边界：考的是"写歪了会不会把正文吃掉"
 *   check_markdown.mjs  真实语料：考的是"真东西有没有被漏掉"
 * **两个都要，因为它们失败的形态不一样**——前者失败是页面上出现乱码，
 * 后者失败是**一切都好好的、只是有几条答案的表格没画出来**。
 *
 * ## 它为什么能进 CI（而 check_ui.mjs 不能）
 *
 * 只依赖 node 和仓库里已有的文件，**不需要起服务、不需要 key、不需要网络**。
 * 跑一遍一秒钟，所以放在单测那个 job 里顺手就跑了。
 *
 * 用法：node tests/ui/check_markdown.mjs
 */
import { api } from "./dom_shim.mjs";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, "..", "..", "fixtures", "replay", "llm");

let failed = 0;
const ok = (cond, msg) => {
  console.log((cond ? "  ✅ " : "  ❌ ") + msg);
  if (!cond) failed++;
};

// ---- 读语料 ----
const texts = [];
for (const f of readdirSync(FIXTURES)) {
  if (!f.endsWith(".json")) continue;
  try {
    const d = JSON.parse(readFileSync(join(FIXTURES, f), "utf-8"));
    const t = d.text || "";
    if (t.trim()) texts.push({ f, t });
  } catch { /* 坏文件跳过：录制中途断掉会留下半个 JSON，那不是这里要管的事 */ }
}
console.log(`\n读到 ${texts.length} 条录制正文`);

// ⚠️ **下面这个断言不是废话，它是这个文件里最有价值的一条。**
// 从 0 条语料上跑出来的"全部通过"是**假绿**——它证不了任何事，
// 只是把"读不到文件"洗成了"没有问题"。CI 上路径写错、fixtures 被
// 挪走、或者以后换成按需下载，都会走到这里。所以先证明语料真的读到了。
ok(texts.length > 100, `语料读到了（${texts.length} 条）——0 条时后面的"全过"没有意义`);

// 真实答案有多长。短的是内部调用（路由/记忆/改写），长的是给用户看的答案。
const long = texts.filter((x) => x.t.length > 400);
console.log(`  其中长正文（>400 字）${long.length} 条，这些才是用户看的那种答案`);
ok(long.length > 0, "语料里有给用户看的长答案（没有的话这个文件测了个寂寞）");

// ---- ① 漏渲染：渲染完还剩行首标记，就是漏了 ----
// 只看**行首**：正文中间合法地出现 `-` `|` `*` 太正常了，
// 拿它们当"漏了"会得到一堆假警报——而**假警报多的检查会被人关掉**。
const LEAK = {
  标题: (p) => /^#{1,6}\s/m.test(p),
  加粗: (p) => /\*\*/.test(p),
  分割线: (p) => /^\s*[-*_]{3,}\s*$/m.test(p),
  表格: (p) => /^\s*\|.*\|/m.test(p),
  无序列表: (p) => /^\s*[-*+]\s+/m.test(p),
  有序列表: (p) => /^\s*\d+[.)]\s+/m.test(p),
};
console.log("\n① 漏渲染（每种都应当是 0 条）");
for (const [name, pred] of Object.entries(LEAK)) {
  const bad = [];
  for (const { f, t } of texts) {
    // 剥掉标签再找：留在纯文字里的标记才是真的露出来了
    const plain = api.mdToHtml(t, [1, 2, 3, 4, 5, 6, 7, 8, 9]).replace(/<[^>]+>/g, "\n");
    if (pred(plain)) bad.push(f);
  }
  ok(bad.length === 0, `${name}：0 条残留${bad.length ? "，实际 " + bad.length + " 条，如 " + bad[0] : ""}`);
}

// ---- ② 反向覆盖：源文里有这个构造的，都渲染出对应元素了吗 ----
// ⚠️ **只有 ① 是不够的。** ① 数的是"还剩多少没吃掉"，
// 而有一种失败它完全看不见：某个构造**压根没走进渲染路径**。
// 那时 ① 里这个构造当然永远是 0 条残留（因为标记压根没被当成标记），
// 于是检查全绿，而表格从来没画出来过。
// 所以反着再数一遍：源文里几篇用了它 → 渲染后几篇有产物。**两个数必须相等。**
const COVER = [
  { name: "标题", src: /\n?#{1,6}\s/m, out: /<h[345]>/ },
  { name: "无序列表", src: /^[ \t]*[-*+]\s+/m, out: /<ul>/ },
  { name: "有序列表", src: /^[ \t]*\d+[.)]\s+/m, out: /<ol>/ },
  { name: "表格", src: /^[ \t]*\|/m, out: /<table>/ },
  { name: "分割线", src: /^[ \t]*([-*_])\1{2,}[ \t]*$/m, out: /<hr>/ },
  { name: "加粗", src: /\*\*[^*\n]+\*\*/, out: /<strong>/ },
];
console.log("\n② 构造 → 元素 的覆盖（源文篇数 = 渲染出篇数）");
for (const { name, src, out } of COVER) {
  let s = 0, o = 0;
  const miss = [];
  for (const { f, t } of texts) {
    const has = src.test(t);
    const got = out.test(api.mdToHtml(t, [1, 2, 3, 4, 5, 6, 7, 8, 9]));
    if (has) s++;
    if (got) o++;
    if (has && !got) miss.push(f);
  }
  ok(s === o, `${name}：源文 ${s} 篇 → 渲染出 ${o} 篇${miss.length ? "，漏了 " + miss[0] : ""}`);
}

// ---- ③ 标签配平 ----
// 浏览器对不配平的标签是**容错**的（会自动补），所以页面上看不出错，
// 但补在哪里由浏览器猜——表现在页面上就是"后半篇莫名其妙都变粗了"。
console.log("\n③ 标签配平");
let unbalanced = 0, worst = null;
for (const { f, t } of texts) {
  const h = api.mdToHtml(t, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  for (const tag of ["p", "ul", "ol", "li", "table", "blockquote", "strong", "code", "span"]) {
    const o = (h.match(new RegExp(`<${tag}[ >]`, "g")) || []).length;
    const c = (h.match(new RegExp(`</${tag}>`, "g")) || []).length;
    if (o !== c) { unbalanced++; worst = `${f}: <${tag}> ${o} 开 / ${c} 闭`; }
  }
}
ok(unbalanced === 0, `所有标签都配平${unbalanced ? "，实际 " + unbalanced + " 处，如 " + worst : ""}`);

// ---- ④ 块级元素不能互相嵌套错（<p> 里塞 <div>/<ul> 会被浏览器拆开）----
// 这是块级解析最容易出的错：段落里混进一个列表行，如果把整个缓冲区
// 一起包进 <p>，浏览器会把 <p> 强行拆成两半，页面结构就和预期不一样了。
console.log("\n④ 块级嵌套");
let nested = 0, sample = null;
for (const { f, t } of texts) {
  const h = api.mdToHtml(t, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  // 粗暴但够用：<p> ... </p> 之间不该出现任何会另起块的标签
  for (const m of h.matchAll(/<p>([\s\S]*?)<\/p>/g)) {
    if (/<(ul|ol|li|table|blockquote|h[1-6]|div|hr)\b/.test(m[1])) { nested++; sample = f; }
  }
}
ok(nested === 0, `没有块级元素被塞进 <p> 里${nested ? "，实际 " + nested + " 处，如 " + sample : ""}`);

// ---- ⑤ 引用标记：后端写 `[1]`，正文里必须变成可点的 ----
console.log("\n⑤ 引用标记");
const withCites = texts.filter((x) => /\[\d{1,2}\]/.test(x.t));
ok(withCites.length > 0, `语料里有 ${withCites.length} 条带 [N] 引用标记的`);
// 全部编号都当合法，这时**每一个** [N] 都该变成可点标记。
// 这条要是挂了，等于"提示词要求的引用写法，页面上不认"——那正是这次的病根。
//
// ⚠️ 这里的写法有个坑，第一版就踩了：不能拿"剥掉标签后还剩没剩 [N]"来判断。
// 可点标记长这样 `<span class="cite-ref">[1]</span>`——**它自己的正文就是 `[1]`**，
// 剥掉标签当然又看见 `[1]` 了，于是 5 条语料全被判成"没转换"。**假红。**
// 正确做法是先把整个 span 摘干净，再看剩下还有没有 `[N]`。
let unconverted = 0, first = null;
for (const { f, t } of withCites) {
  const h = api.mdToHtml(t, [...Array(20).keys()]);
  const plain = h
    .replace(/<span class="cite-ref"[^>]*>[\s\S]*?<\/span>/g, "")
    .replace(/<[^>]+>/g, "");
  if (/\[\d{1,2}\]/.test(plain)) { unconverted++; if (!first) first = f; }
}
ok(unconverted === 0, `[1] 全部被渲染成可点标记${unconverted ? "，实际漏了 " + unconverted + " 条，如 " + first : ""}`);

// 再按**个数**对一次账，比"有没有漏"更严：两种写法混在一句话里时要都对。
// （`[1,2]` 会变成两个 span，所以这里只对单编号的条数做下限检查。）
let src1 = 0, got1 = 0;
for (const { t } of withCites) {
  src1 += (t.match(/\[\d{1,2}\]/g) || []).length;
  got1 += (api.mdToHtml(t, [...Array(20).keys()]).match(/class="cite-ref"/g) || []).length;
}
ok(got1 >= src1, `可点标记个数 ${got1} ≥ 源文引用个数 ${src1}`);

// ---- ⑥ 流式：**答案是一批一批到的，每一批中间态都得能看** ----
// ⚠️ 这一组测的是这次改动风险最大的一处：答案区从"追加纯文本"改成了
// "整段重渲染"，而整段重渲染意味着**中间状态也会被用户看到**。
// 模型吐 `**加粗**` 是一个字一个字来的，吐到第三个字符时屏幕上是
// 一个孤零零的 `**`——写歪了的话，从那里到下一个 `**` 之间
// 半篇答案都会变成粗体，然后再弹回来。**用户看到的就是这个。**
//
// 所以这里把真实答案**逐字**喂一遍，每一步都检查：
//   ① 没有残留的 `**`（没配对的被 streamSafe 藏了）
//   ② 标签是配平的（半截标签会让浏览器猜到别处去）
// 逐字跑一整篇是最笨也最直接的办法，而且它便宜——最长的一篇才 1121 字。
console.log("\n⑥ 流式中间态（逐字喂真实答案）");
const VALID = [...Array(20).keys()];
let stepBad = 0, stepFirst = null;
const longest = long.reduce((a, b) => (b.t.length > a.t.length ? b : a));
for (let n = 1; n <= longest.t.length; n++) {
  const h = api.mdToHtml(api.streamSafe(longest.t.slice(0, n)), VALID);
  const plain = h.replace(/<span class="cite-ref"[^>]*>[\s\S]*?<\/span>/g, "").replace(/<[^>]+>/g, "");
  const unbalanced = ["p", "ul", "ol", "li", "table", "blockquote", "strong", "code"]
    .some((tag) =>
      (h.match(new RegExp(`<${tag}[ >]`, "g")) || []).length !==
      (h.match(new RegExp(`</${tag}>`, "g")) || []).length);
  if (/\*\*/.test(plain) || unbalanced) {
    stepBad++;
    if (!stepFirst) stepFirst = `第 ${n} 字: ${JSON.stringify(longest.t.slice(0, n).slice(-40))}`;
  }
}
ok(stepBad === 0,
   `逐字喂完 ${longest.t.length} 个中间态，全部能看${stepBad ? "，实际 " + stepBad + " 个不行，首个 " + stepFirst : ""}`);

// 每个前缀都合法还不够，得确认**最后那一步是完整的**——
// 否则上面那条在每个字符都"干净"（比如渲染器什么都不认）时会假绿。
const full = api.mdToHtml(longest.t, VALID);
ok(/<strong>/.test(full) && /<h3>|<\/h3>/.test(full),
   "同一篇走完全程后，加粗和标题都还在（不是靠什么都不认来达标）");

console.log(failed ? `\n❌ ${failed} 项没过` : "\n✅ 全部通过");
process.exit(failed ? 1 : 0);
