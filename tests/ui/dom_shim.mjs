/* 用最小的 DOM 桩把 index.html 里的脚本真的跑起来，然后拿**真 schema** 和
 * **真 payload** 喂给它。
 *
 * 为什么值得搭这个：`node --check` 只能证明"语法对"。
 * 我在这份页面里已经犯过一次语法完全正确、一跑就炸的错
 * （引了一个不存在的对象）。而表单生成器的分支很多
 * （枚举/可空布尔/日期/数字/数组/兜底），只在浏览器里手点很难盖全。
 *
 * 这不是替代浏览器测试，是**在打开浏览器之前**先把逻辑层面的错挡掉。
 */
import fs from "node:fs";

class El {
  constructor(tag = "div") {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this._cls = "";
    this._html = "";
    this._text = "";
    this.style = {};
    this.dataset = {};
    this.hidden = false;
    this.disabled = false;
    this.title = "";
    this.value = "";
    this.open = false;
    this._listeners = {};
    const self = this;
    this.classList = {
      add: (...c) => { self._cls = [...new Set([...self._cls.split(" ").filter(Boolean), ...c])].join(" "); },
      remove: (...c) => { self._cls = self._cls.split(" ").filter(x => x && !c.includes(x)).join(" "); },
      toggle: (c, force) => {
        const has = self._cls.split(" ").includes(c);
        const want = force === undefined ? !has : force;
        want ? self.classList.add(c) : self.classList.remove(c);
      },
      contains: (c) => self._cls.split(" ").includes(c),
    };
  }
  get className() { return this._cls; }
  set className(v) { this._cls = v || ""; }
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = String(v); this.children = []; }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); }
  appendChild(n) { n.parentNode = this; this.children.push(n); return n; }
  addEventListener(ev, fn) { (this._listeners[ev] ||= []).push(fn); }
  click() { (this._listeners.click || []).forEach(f => f({ target: this })); }
  scrollIntoView() {}
  _all() {
    const out = [];
    for (const c of this.children) { out.push(c, ...c._all()); }
    return out;
  }
  querySelector(sel) { return this._match(sel)[0] || null; }
  querySelectorAll(sel) { return this._match(sel); }
  _match(sel) {
    // 只支持这个页面真正用到的三种选择器形态。
    const m = sel.match(/^([a-z]*)(?:\.([\w-]+))?(?:\[data-idx="?([^"\]]+)"?\])?$/i);
    if (!m) throw new Error("DOM 桩不支持的选择器：" + sel);
    const [, tag, cls, idx] = m;
    return this._all().filter(n => {
      if (tag && n.tagName !== tag.toUpperCase()) return false;
      if (cls && !n.classList.contains(cls)) return false;
      if (idx && String(n.dataset.idx) !== String(idx)) return false;
      return true;
    });
  }
}

globalThis.document = {
  getElementById: () => new El("div"),
  createElement: (t) => new El(t),
  addEventListener: () => {},
  body: new El("body"),
};
globalThis.window = { addEventListener: () => {}, scrollTo: () => {}, innerHeight: 0, scrollY: 0 };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
// 页面脚本一加载就会自己 fetch /api/health，这里得给它一个假响应，
// 否则它会去真发请求（还可能因为连不上而抛）。
// ⚠️ 但真 fetch 要**留一份**给 check-ui.mjs 用 —— 它得去问真后端。
const realFetch = globalThis.fetch;
globalThis.fetch = async () => ({ ok: false, status: 0, json: async () => ({}) });
// ⚠️ **只补缺，不整个替换。** Node 自带的 `performance` 上还有
// `markResourceTiming` 之类的东西，undici 的 fetch 内部依赖它。
// 直接 `globalThis.performance = {now: ...}` 会让所有 fetch 崩在
// `markResourceTiming is not a function`——而报错栈指向 undici 内部，
// 看起来完全不像是我这行的问题。
if (!globalThis.performance) globalThis.performance = {};
if (!globalThis.performance.now) globalThis.performance.now = () => 0;
globalThis.alert = () => {};
globalThis.Option = class { constructor(t, v) { const e = new El("option"); e.text = t; e.value = v; return e; } };

const html = fs.readFileSync(new URL("../../web/index.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// 把脚本末尾挂一个 return，好把内部函数取出来。
const factory = new Function(
  script + "\n; return { normalize, enumLabels, valueText, schemaToForm, readForm, renderPayload, parseFrame, rich, esc };"
);
const api = factory();

export { api, El, realFetch };
export default api;
