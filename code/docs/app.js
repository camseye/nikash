/* Nikash scoreboard.
 *
 * Reads data/scoreboard.json and data/examples.json and renders the
 * page sections. Log lines on this page are attacker-authored by
 * design, so every string from the data files is inserted with
 * textContent / createTextNode — never innerHTML.
 */
(() => {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';

  const state = {
    board: null,
    examples: null,
    bench: null,
    task: null,
    model: 'avg',
    caseIdx: 0,
    defence: 'none',
    focus: null,
  };

  // ---------- DOM helpers ----------

  const $ = (sel) => document.querySelector(sel);

  function el(tag, attrs, ...kids) {
    const node = document.createElement(tag);
    setAttrs(node, attrs);
    append(node, kids);
    return node;
  }

  function sv(tag, attrs, ...kids) {
    const node = document.createElementNS(SVG_NS, tag);
    setAttrs(node, attrs);
    append(node, kids);
    return node;
  }

  function setAttrs(node, attrs) {
    if (!attrs) return;
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'text') node.textContent = String(value);
      else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : String(value));
    }
  }

  function append(node, kids) {
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  /**
   * Fills a group of toggle buttons, rebuilding them only when `sig` changes.
   * Otherwise only aria-pressed is updated: replacing the clicked button would drop keyboard focus.
   */
  function toggleGroup(host, sig, items, pressed) {
    if (host.dataset.sig !== sig) {
      clear(host);
      host.dataset.sig = sig;
      for (const it of items) host.append(el('button', { type: 'button', 'data-key': it.key, ...it.attrs }, ...(it.kids || [])));
    }
    for (const btn of host.querySelectorAll('button[data-key]')) {
      btn.setAttribute('aria-pressed', btn.dataset.key === pressed ? 'true' : 'false');
    }
  }

  // ---------- Numbers ----------

  const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  const pct = (x) => (isNum(x) ? `${Math.round(x * 100)}%` : '—');
  const TIE = 1e-9;
  /** "36" or "36–196" for a list of counts. */
  const count = (ns) => (Math.min(...ns) === Math.max(...ns) ? String(ns[0]) : `${Math.min(...ns)}–${Math.max(...ns)}`);
  /** "88%" or "45–88%" for a list of rates. */
  const span = (vs) => {
    const lo = Math.round(Math.min(...vs) * 100), hi = Math.round(Math.max(...vs) * 100);
    return lo === hi ? `${lo}%` : `${lo}–${hi}%`;
  };
  const names = (rows) => rows.map((r) => r.d.name).join(', ');

  /** Mean of the defined values; null when there are none. */
  function mean(xs) {
    const vals = xs.filter(isNum);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  }

  function wilson(p, n, z = 1.96) {
    if (!n) return [p, p];
    const denom = 1 + (z * z) / n;
    const centre = (p + (z * z) / (2 * n)) / denom;
    const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
    return [Math.max(0, centre - half), Math.min(1, centre + half)];
  }

  // ---------- Data checks ----------

  class DataError extends Error {
    constructor(kind, message) {
      super(message);
      this.kind = kind;
    }
  }

  const hasIds = (xs) => Array.isArray(xs) && xs.length > 0 && xs.every((x) => x && typeof x === 'object' && typeof x.id === 'string');

  function checkBoard(board) {
    if (!board || typeof board !== 'object') throw new DataError('invalid', 'data/scoreboard.json is not an object');
    for (const key of ['benchmarks', 'tasks', 'models', 'defences']) {
      if (!hasIds(board[key])) throw new DataError('invalid', `data/scoreboard.json: "${key}" is missing or malformed`);
    }
    if (!board.cells || typeof board.cells !== 'object') throw new DataError('invalid', 'data/scoreboard.json: "cells" is missing');
  }

  const unit = (v) => isNum(v) && v >= 0 && v <= 1;
  const MATRICES = {
    asr: unit,
    cost: unit,
    asr_ci: (v) => Array.isArray(v) && v.length === 2 && unit(v[0]) && unit(v[1]) && v[0] <= v[1],
    asr_n_eff: (v) => isNum(v) && v >= 0,
    cost_parse_as_error: unit,
    miss: unit,
    false_alarm: unit,
    parse_fail: unit,
    kept_share: unit,
    asr_n: (v) => Number.isInteger(v) && v >= 0,
  };

  /**
   * Reshapes every cell matrix to defences × models. Malformed entries become null
   * (shown as a dash); the keys of cells that needed fixing are returned.
   */
  function normaliseCells(board) {
    const nDef = board.defences.length;
    const nMod = board.models.length;
    const flagged = new Set();
    for (const [key, c] of Object.entries(board.cells)) {
      if (!c || typeof c !== 'object' || !Array.isArray(c.asr) || !Array.isArray(c.cost)) {
        delete board.cells[key];
        flagged.add(key);
        continue;
      }
      for (const [name, valid] of Object.entries(MATRICES)) {
        const src = c[name];
        if (src === undefined) continue;
        if (!Array.isArray(src)) {
          delete c[name];
          flagged.add(key);
          continue;
        }
        if (src.length !== nDef) flagged.add(key);
        c[name] = Array.from({ length: nDef }, (_, di) => {
          const row = Array.isArray(src[di]) ? src[di] : [];
          if (row.length !== nMod) flagged.add(key);
          return Array.from({ length: nMod }, (_, mi) => {
            const v = row[mi];
            if (v === null || v === undefined) return null;
            if (valid(v)) return v;
            flagged.add(key);
            return null;
          });
        });
      }
    }
    return [...flagged];
  }

  // ---------- Data access ----------

  const defences = () => state.board.defences;
  const models = () => state.board.models;
  const benchmarks = () => state.board.benchmarks;
  const benchById = (id) => benchmarks().find((b) => b.id === id);
  const taskById = (id) => state.board.tasks.find((t) => t.id === id);
  const modelIndex = () => models().findIndex((m) => m.id === state.model);
  const modelName = () => (modelIndex() < 0 ? 'average of all models' : models()[modelIndex()].name);
  // Only scoreboard.json decides this; a missing flag counts as illustrative.
  const isIllustrative = () => state.board.illustrative !== false;

  function cell(bench, task) {
    return state.board.cells[`${bench}|${task}`] || null;
  }

  function intervalFor(p, ci, n) {
    if (!isNum(p)) return null;
    if (ci) return ci;
    return isNum(n) && n > 0 ? wilson(p, n) : null;
  }

  /** One defence's numbers on one benchmark/task for the selected model. */
  function scoreFor(bench, task, defIdx) {
    const c = cell(bench, task);
    if (!c) return null;
    const asrRow = c.asr[defIdx];
    const costRow = c.cost[defIdx];
    const ciRow = c.asr_ci?.[defIdx];
    const sizeRow = c.asr_n_eff?.[defIdx];
    const fallbackN = state.board.n?.[bench]?.adversarial;
    const sizeOf = (mi) => (sizeRow ? sizeRow[mi] : fallbackN);
    const perCi = asrRow.map((p, mi) => intervalFor(p, ciRow?.[mi], sizeOf(mi)));
    const mi = modelIndex();
    const extra = (name) => c[name]?.[defIdx] || [];
    const pick = (name) => (mi < 0 ? mean(extra(name)) : extra(name)[mi] ?? null);
    const detail = {
      miss: pick('miss'), falseAlarm: pick('false_alarm'), parseFail: pick('parse_fail'),
      keptN: (mi < 0 ? extra('asr_n') : [extra('asr_n')[mi]]).filter(isNum),
    };
    if (mi < 0) {
      const asr = mean(asrRow);
      // Every model answers the same windows, so pooling models adds no independent cases.
      const sizes = asrRow.map((p, i) => (isNum(p) ? sizeOf(i) : null)).filter((n) => isNum(n) && n > 0);
      const ci = isNum(asr) && sizes.length ? wilson(asr, Math.min(...sizes)) : null;
      return { asr, cost: mean(costRow), ci, perModel: asrRow, perCi, detail };
    }
    return { asr: asrRow[mi], cost: costRow[mi], ci: perCi[mi], perModel: asrRow, perCi, detail };
  }

  /** Defences by attack success (cost breaks ties in the order only); exact ties share a rank. */
  function ranked(bench, task) {
    const key = (x) => (isNum(x) ? x : Infinity);
    const cmp = (a, b) => (key(a) === key(b) ? 0 : key(a) - key(b));
    const rows = defences()
      .map((d, i) => ({ def: d, idx: i, score: scoreFor(bench, task, i), rank: null }))
      .filter((r) => r.score)
      .sort((a, b) => cmp(a.score.asr, b.score.asr) || cmp(a.score.cost, b.score.cost));
    rows.forEach((r, i) => {
      if (!isNum(r.score.asr)) return;
      const prev = rows[i - 1];
      r.rank = prev && isNum(prev.score.asr) && Math.abs(prev.score.asr - r.score.asr) < TIE ? prev.rank : i + 1;
    });
    return rows;
  }

  // ---------- Status ribbon ----------

  function showRibbon(kind, label, text) {
    const ribbon = $('#ribbon');
    if (kind === 'error') ribbon.classList.add('error');
    ribbon.append(el('p', null, el('strong', { text: label }), text));
    ribbon.hidden = false;
  }

  // ---------- Hero tally ----------

  function renderTally() {
    const b = state.board;
    const items = [
      [b.benchmarks.length, 'benchmarks', b.benchmarks.map((x) => x.family).join(' · ')],
      [b.defences.length, 'defences', 'prompt, filter and architectural'],
      [b.models.length, 'open-weight models', 'self-hosted, 4-bit'],
      [b.tasks.length, 'tasks', b.tasks.map((t) => String(t.name).toLowerCase()).join(' and ')],
    ];
    const tally = $('#tally');
    clear(tally);
    for (const [num, label, sub] of items) {
      tally.append(el('div', null, el('dt', { text: label }), el('dd', null, String(num), el('small', { text: sub }))));
    }
    $('#updated').textContent = b.updated || '—';
    const nMod = b.models.length;
    const hint = $('#spread-hint');
    clear(hint);
    hint.append(`attack success, ${nMod === 1 ? 'one model' : `${nMod} models`}`);
    const key = el('span', { class: 'model-key' });
    b.models.forEach((m, i) => {
      const glyph = sv('svg', { viewBox: '0 0 12 12', 'aria-hidden': 'true' }, marker(shapeOf(i), 6, 6, 4, { class: 'mk' }));
      key.append(el('span', { title: m.name }, glyph, shortModel(m.name)));
    });
    hint.append(key);
  }

  // ---------- Wall: attack fragment and defence gradient ----------

  function logLine(line) {
    const row = el('span', { class: 'line' });
    for (const seg of line) {
      if (seg.a) row.append(el('mark', { class: 'ac', title: `Attacker-controlled field: ${seg.field || 'unknown'}`, text: seg.t }));
      else row.append(document.createTextNode(seg.t ?? ''));
    }
    return row;
  }

  function renderFragment() {
    const host = $('#fragment');
    clear(host);
    const c = state.examples.cases[0];
    if (!c) return;
    const lines = c.lines.filter((l) => !(l.length === 1 && l[0].ellipsis));
    const withPayload = lines.findIndex((l) => l.some((s) => s.a));
    const start = Math.max(0, Math.min(withPayload, lines.length - 2));
    for (const line of lines.slice(start, start + 2)) host.append(logLine(line), document.createTextNode('\n'));
  }

  function renderGradient() {
    const host = $('#gradient');
    clear(host);
    const rows = ranked(state.bench, state.task);
    rows.slice().reverse().forEach((r) => {
      const fill = el('span', { class: 'fill' });
      fill.style.width = `${((isNum(r.score.asr) ? r.score.asr : 0) * 100).toFixed(1)}%`;
      host.append(el('li', { class: `d-${r.def.id}${r.rank === 1 ? ' best' : ''}` },
        el('span', { class: 'name', text: r.def.name }),
        el('span', { class: 'track' }, fill),
        el('b', { text: pct(r.score.asr) })));
    });
    const note = $('#gradient-note');
    clear(note);
    append(note, ['Attack success, following the ', el('a', { href: '#scoreboard', text: 'scoreboard' }),
      ` selection: ${benchById(state.bench).name} · ${taskById(state.task).name} · ${modelName()}.`]);
  }

  // ---------- §1 Anatomy ----------

  function renderCaseTabs() {
    const items = state.examples.cases.map((c, i) => ({
      key: String(i),
      attrs: { class: 'case-tab', text: c.title, onclick: () => { state.caseIdx = i; renderAnatomy(); } },
    }));
    toggleGroup($('#case-tabs'), 'cases', items, String(state.caseIdx));
  }

  function renderAnatomy() {
    renderCaseTabs();
    const c = state.examples.cases[state.caseIdx];
    if (!c) return;

    const meta = $('#case-meta');
    clear(meta);
    append(meta, [
      el('span', null, el('b', { text: 'Technique ' }), c.technique),
      el('span', null, el('b', { text: 'Source ' }), c.source),
      el('span', null, el('b', { text: 'Benchmark ' }), c.benchmark),
      el('span', null, el('b', { text: 'Task ' }), taskById(c.task)?.name ?? c.task),
    ]);

    const log = $('#case-log');
    clear(log);
    for (const line of c.lines) {
      const row = logLine(line);
      if (line.length === 1 && line[0].ellipsis) row.classList.add('ellipsis');
      log.append(row, document.createTextNode('\n'));
    }

    const truth = $('#case-truth');
    clear(truth);
    append(truth, [el('b', { text: 'Ground truth. ' }), c.truth]);

    renderChips(c);
    renderVerdict(c);
  }

  function renderChips(c) {
    const items = defences().filter((d) => c.outcomes[d.id]).map((d) => {
      const out = c.outcomes[d.id];
      return {
        key: d.id,
        attrs: { class: `chip d-${d.id}`, onclick: () => { state.defence = d.id; renderChips(c); renderVerdict(c); } },
        kids: [d.name, el('span', { class: `chip-state ${out.held ? 'held' : 'breach'}`, text: out.held ? '✓' : '✗', 'aria-label': out.held ? 'held' : 'breached' })],
      };
    });
    toggleGroup($('#def-chips'), `case-${state.caseIdx}`, items, state.defence);
  }

  function renderVerdict(c) {
    const card = $('#verdict-card');
    clear(card);
    const d = defences().find((x) => x.id === state.defence) || defences()[0];
    const out = c.outcomes[d.id];
    if (!out) {
      card.append(el('p', { class: 'verdict-def', text: 'No response recorded for this defence.' }));
      return;
    }
    const isSummary = c.task === 'summary';
    card.append(
      el('div', { class: 'verdict-top' },
        el('span', { class: `badge ${out.held ? 'held' : 'breach'}`, text: out.held ? 'Defence held' : 'Attack succeeded' }),
        el('span', { class: 'verdict-label', text: isSummary ? 'Summary written by the assistant' : `Verdict: ${out.verdict}` })),
      el('p', { class: 'verdict-def', text: `${d.name}: ${d.short}` }),
      el('blockquote', { text: out.explanation }),
    );
    if (out.consistent === false) {
      card.append(el('p', { class: 'flag', text: 'Explanation check: the reasoning points one way and the verdict the other. Nikash flags answers like this.' }));
    }
  }

  // ---------- §2 Controls and scoreboard ----------

  function renderControls() {
    toggleGroup($('#ctl-bench'), 'benchmarks', benchmarks().map((b) => ({
      key: b.id,
      attrs: { title: `${b.family} · ${b.licence}`, onclick: () => { state.bench = b.id; renderResults(); } },
      kids: [`${b.short} · ${b.name}`],
    })), state.bench);

    toggleGroup($('#ctl-task'), 'tasks', state.board.tasks.map((t) => ({
      key: t.id,
      attrs: { text: t.name, onclick: () => { state.task = t.id; renderResults(); } },
    })), state.task);

    const sel = $('#ctl-model');
    if (!sel.options.length) {
      sel.append(el('option', { value: 'avg', text: 'Average of all models' }));
      for (const m of models()) sel.append(el('option', { value: m.id, text: m.note ? `${m.name} (${m.note})` : m.name }));
      sel.addEventListener('change', () => { state.model = sel.value; renderResults(); });
    }
    sel.value = state.model;
  }

  const ciText = (ci) => `95% interval ${pct(ci[0])}–${pct(ci[1])}`;

  function meter(kind, value, ci, defId) {
    const track = el('span', { class: 'meter-track' });
    if (isNum(value)) {
      const fill = el('span', { class: 'meter-fill' });
      fill.style.width = `${(value * 100).toFixed(1)}%`;
      track.append(fill);
    }
    if (ci) {
      const whisker = el('span', { class: 'meter-ci', title: ciText(ci) });
      whisker.style.left = `${(ci[0] * 100).toFixed(1)}%`;
      whisker.style.width = `${Math.max(0.5, (ci[1] - ci[0]) * 100).toFixed(1)}%`;
      track.append(whisker);
    }
    return el('div', { class: `meter ${kind} d-${defId}` }, el('span', { class: 'meter-val', text: pct(value) }), track);
  }

  // One marker shape per model, so a model can be told apart without relying on colour.
  const SHAPES = ['circle', 'square', 'triangle', 'diamond'];
  const shapeOf = (i) => SHAPES[i % SHAPES.length];
  const shortModel = (name) => String(name).split('-').slice(0, 2).join('-');

  function marker(shape, cx, cy, r, attrs) {
    if (shape === 'circle') return sv('circle', { cx: cx.toFixed(1), cy: cy.toFixed(1), r, ...attrs });
    const pts = {
      square: [[-0.85, -0.85], [0.85, -0.85], [0.85, 0.85], [-0.85, 0.85]],
      triangle: [[0, -1.15], [1.05, 0.8], [-1.05, 0.8]],
      diamond: [[0, -1.2], [1.2, 0], [0, 1.2], [-1.2, 0]],
    }[shape];
    const d = `M${pts.map(([a, b]) => `${(cx + a * r).toFixed(1)},${(cy + b * r).toFixed(1)}`).join(' L')} Z`;
    return sv('path', { d, ...attrs });
  }

  function spread(perModel, perCi, defId) {
    const w = 160, pad = 7, mid = 15, r = 5, lanes = [0, -10, 10];
    const x = (v) => pad + v * (w - 2 * pad);
    const label = (v, i) => `${models()[i].name}: ${pct(v)}${perCi[i] ? ` (${ciText(perCi[i])})` : ''}`;
    const g = sv('svg', { viewBox: `0 0 ${w} 41`, role: 'img', 'aria-label': perModel.map(label).join(', ') });
    g.append(sv('line', { class: 'axis', x1: pad, x2: w - pad, y1: mid, y2: mid }));
    // Markers less than two radii apart move to a free lane above or below the axis.
    const placed = [];
    perModel
      .map((v, i) => ({ v, i }))
      .filter((p) => isNum(p.v))
      .sort((a, b) => a.v - b.v)
      .forEach(({ v, i }) => {
        const cx = x(v);
        const busy = placed.filter((p) => Math.abs(p.cx - cx) < 2 * r).map((p) => p.lane);
        const lane = Math.max(0, lanes.findIndex((_, k) => !busy.includes(k)));
        placed.push({ cx, lane });
        const mk = marker(shapeOf(i), cx, mid + lanes[lane], r, { class: 'mk' });
        mk.append(sv('title', { text: label(v, i) }));
        g.append(mk);
      });
    g.append(sv('text', { x: pad, y: 39, text: '0' }), sv('text', { x: w - pad, y: 39, 'text-anchor': 'end', text: '100%' }));
    return el('div', { class: `spread d-${defId}` }, g);
  }

  function sampleNote(n) {
    if (!n) return '';
    const counts = [isNum(n.injected) && `${n.injected} injected`, isNum(n.clean) && `${n.clean} clean`].filter(Boolean);
    const parts = [];
    if (counts.length) parts.push(`Cases per model (smallest across models): ${counts.join(', ')}`);
    if (isNum(n.adversarial)) {
      parts.push(`${counts.length ? 'smallest' : 'Smallest'} effective injected sample, after twin-conditioning and clustering: ${n.adversarial}`);
    }
    return parts.length ? `${parts.join('; ')}. ` : '';
  }

  function renderBoard() {
    const body = $('#board-body');
    clear(body);
    const rows = ranked(state.bench, state.task);
    const b = benchById(state.bench);
    $('#board-caption').textContent = `Defence scoreboard for ${b.name}, ${taskById(state.task).name}, ${modelName()}`;

    if (!rows.length) {
      body.append(el('tr', null, el('td', { colspan: 5, text: 'No results for this combination yet.' })));
    }

    const nInjected = state.board.n?.[state.bench]?.injected;
    for (const r of rows) {
      const { def, score } = r;
      const { miss, falseAlarm, keptN } = score.detail;
      const keptText = keptN.length ? `on ${count(keptN)}${isNum(nInjected) ? ` of ${nInjected}` : ''} attacks` : null;
      const errText = isNum(miss) || isNum(falseAlarm) ? `misses ${pct(miss)} · false alarms ${pct(falseAlarm)}` : null;
      body.append(el('tr', { class: r.rank === 1 ? 'top' : null },
        el('td', null, el('span', { class: 'rank', text: r.rank ? String(r.rank) : '—' })),
        el('td', null,
          el('div', { class: `def-name d-${def.id}` }, def.name, el('span', { class: 'tag kind', text: def.kind })),
          el('div', { class: 'def-short', text: def.short })),
        el('td', null, meter('asr', score.asr, score.ci, def.id), keptText && el('div', { class: 'meter-sub', text: keptText })),
        el('td', null, meter('cost', score.cost, null, def.id), errText && el('div', { class: 'meter-sub', text: errText })),
        el('td', null, spread(score.perModel, score.perCi, def.id)),
      ));
    }
    renderCaveats();

    const task = taskById(state.task);
    $('#board-note').textContent =
      `Attack succeeds when ${task.attack_goal}. ${sampleNote(state.board.n?.[state.bench])}` +
      `Benchmark licence: ${b.licence}. Source: ${b.source}.`;
  }

  /** Thresholds above which a number is called out in "Read with care"; index.html states them in .caveats-foot. */
  const CAVEAT = { falseAlarm: 0.5, miss: 0.5, kept: 0.5, parseFail: 0.05 };

  /** Notes for the scoreboard selection where a low attack success needs context. */
  function caveats(bench, task) {
    const c = cell(bench, task);
    if (!c) return [];
    const defs = defences();
    const all = models().map((_, i) => i);
    const mis = modelIndex() < 0 ? all : [modelIndex()];
    const at = (name, di, mi) => c[name]?.[di]?.[mi];
    const nInjected = state.board.n?.[bench]?.injected;
    // A defence whose numbers are identical for every model does not depend on the model.
    const modelFree = (di) => all.length > 1 && ['asr', 'miss', 'false_alarm'].every((k) => {
      const row = c[k]?.[di];
      return Array.isArray(row) && row.every((v) => v === row[0]);
    });
    const out = [];

    const ofAll = isNum(nInjected) ? ` of ${nInjected}` : '';

    defs.forEach((d, di) => {
      if (!modelFree(di)) return;
      const miss = at('miss', di, 0), fa = at('false_alarm', di, 0), kept = at('asr_n', di, 0);
      const parts = [];
      if (isNum(miss) && miss >= CAVEAT.miss) parts.push(`misses ${pct(miss)} of malicious clean windows`);
      if (isNum(fa) && fa >= CAVEAT.falseAlarm) parts.push(`raises false alarms on ${pct(fa)} of benign clean windows`);
      if (!parts.length) return;
      const sample = isNum(kept) && at('kept_share', di, 0) < CAVEAT.kept ? ` Its attack success is measured on only ${kept}${ofAll} attacks.` : '';
      out.push(`${d.name} gives the same answers for every model and ${parts.join(' and ')}.${sample}`);
    });

    for (const mi of mis) {
      const mName = models()[mi].name;
      const over = (name, test) => defs
        .map((d, di) => ({ d, v: at(name, di, mi), di }))
        .filter(({ v, di }) => !modelFree(di) && isNum(v) && test(v));
      const fa = over('false_alarm', (v) => v >= CAVEAT.falseAlarm);
      const miss = over('miss', (v) => v >= CAVEAT.miss);
      const kept = over('kept_share', (v) => v < CAVEAT.kept).filter(({ di }) => isNum(at('asr_n', di, mi)));
      const sentences = [];
      if (fa.length) {
        sentences.push(`${mName} raises false alarms on ${span(fa.map((x) => x.v))} of benign clean windows (${names(fa)}); a model that flags nearly everything is hard to talk into closing an alert, so read its attack success together with this cost.`);
      }
      if (miss.length) {
        sentences.push(`${sentences.length ? 'It also' : mName} misses ${span(miss.map((x) => x.v))} of malicious clean windows (${names(miss)}).`);
      }
      if (kept.length) {
        sentences.push(`${sentences.length ? 'Its' : `For ${mName},`} attack success rests on only ${count(kept.map(({ di }) => at('asr_n', di, mi)))}${ofAll} attacks (${names(kept)}).`);
      }
      if (sentences.length) out.push(sentences.join(' '));
    }

    defs.forEach((d, di) => {
      const bad = mis.map((mi) => ({ mi, v: at('parse_fail', di, mi) })).filter(({ v }) => isNum(v) && v >= CAVEAT.parseFail);
      if (!bad.length) return;
      const who = mis.length > 1 ? ` (${bad.map(({ mi }) => shortModel(models()[mi].name)).join(', ')})` : '';
      const costAll = mean(mis.map((mi) => at('cost_parse_as_error', di, mi)));
      out.push(`${d.name}: ${span(bad.map((x) => x.v))} of answers could not be read${who}. They are left out of attack success and clean-log cost${isNum(costAll) ? `; counting them as errors puts the clean-log cost at ${pct(costAll)}` : ''}.`);
    });
    return out;
  }

  function renderCaveats() {
    const list = $('#caveats-list');
    clear(list);
    const notes = caveats(state.bench, state.task);
    for (const text of notes) list.append(el('li', { text }));
    $('#board-caveats').hidden = notes.length === 0;
  }

  // ---------- §3 Frontier scatter ----------

  function renderFrontier() {
    const svg = $('#frontier-svg');
    for (const child of [...svg.childNodes]) {
      if (child.nodeName !== 'title' && child.nodeName !== 'desc') svg.removeChild(child);
    }
    const W = 760, H = 460, m = { t: 24, r: 28, b: 56, l: 64 };
    const rows = ranked(state.bench, state.task).filter((r) => isNum(r.score.asr) && isNum(r.score.cost));
    const anchors = (Array.isArray(state.board.anchors?.[state.task]) ? state.board.anchors[state.task] : [])
      .filter((a) => a && unit(a.asr) && isNum(a.cost));
    const maxCost = Math.max(0.35, ...rows.map((r) => r.score.cost), ...anchors.map((a) => a.cost)) * 1.08;
    const x = (v) => m.l + (v / maxCost) * (W - m.l - m.r);
    const y = (v) => H - m.b - v * (H - m.t - m.b);

    if (isIllustrative()) {
      svg.append(sv('text', { class: 'watermark', x: W / 2, y: H / 2 + 20, 'text-anchor': 'middle', text: 'ILLUSTRATIVE' }));
    }

    svg.append(sv('rect', { class: 'ideal', x: m.l, y: y(0.2), width: x(maxCost * 0.32) - m.l, height: y(0) - y(0.2) }));
    const better = sv('text', { class: 'ideal-label', x: m.l + 8, y: y(0.2) + 16, text: 'BETTER' });
    svg.append(better);

    const grid = sv('g', { class: 'grid' });
    const ax = sv('g', { class: 'ax' });
    for (let v = 0; v <= 1.0001; v += 0.25) {
      grid.append(sv('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
      ax.append(sv('text', { x: m.l - 10, y: y(v) + 4, 'text-anchor': 'end', text: pct(v) }));
    }
    const step = maxCost > 0.45 ? 0.1 : 0.05;
    for (let v = 0; v <= maxCost + 1e-9; v += step) {
      grid.append(sv('line', { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }));
      ax.append(sv('text', { x: x(v), y: H - m.b + 18, 'text-anchor': 'middle', text: pct(v) }));
    }
    ax.append(sv('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    ax.append(sv('line', { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b }));
    ax.append(sv('text', { class: 'ax-title', x: (m.l + W - m.r) / 2, y: H - 12, 'text-anchor': 'middle', text: 'Clean-log cost (balanced error on clean events) →' }));
    ax.append(sv('text', { class: 'ax-title', x: -(m.t + H - m.b) / 2, y: 18, transform: 'rotate(-90)', 'text-anchor': 'middle', text: 'Attack success →' }));
    svg.append(grid, ax);

    const byCost = [...rows].sort((a, b) => a.score.cost - b.score.cost || a.score.asr - b.score.asr);
    const front = [];
    let best = Infinity;
    for (const r of byCost) {
      if (r.score.asr < best) { front.push(r); best = r.score.asr; }
    }
    if (front.length > 1) {
      const d = front.map((r, i) => `${i ? 'L' : 'M'}${x(r.score.cost).toFixed(1)},${y(r.score.asr).toFixed(1)}`).join(' ');
      svg.append(sv('path', { class: 'pareto', d }));
    }

    const marks = [];
    const anchorLabels = [];
    for (const a of anchors) {
      const cx = x(a.cost), cy = y(a.asr);
      const g = sv('g', { class: 'anchor' });
      g.append(sv('path', { d: `M${cx},${cy - 7} L${cx + 7},${cy} L${cx},${cy + 7} L${cx - 7},${cy} Z` }));
      svg.append(g);
      marks.push({ x0: cx - 9, y0: cy - 9, x1: cx + 9, y1: cy + 9, w: 2 });
      anchorLabels.push({ g, cx, cy, text: a.name, top: a.asr > 0.5 });
    }

    const pointLabels = [];
    for (const r of rows) {
      const { def, score } = r;
      const cx = x(score.cost), cy = y(score.asr);
      const g = sv('g', { class: `pt d-${def.id}` });
      if (score.ci) {
        g.append(sv('line', { x1: cx, x2: cx, y1: y(score.ci[0]), y2: y(score.ci[1]) }));
        marks.push({ x0: cx - 3, y0: y(score.ci[1]), x1: cx + 3, y1: y(score.ci[0]), w: 0.3 });
      }
      const dot = sv('circle', { cx, cy, r: 7 });
      dot.append(sv('title', { text: `${def.name}: attack success ${pct(score.asr)}${score.ci ? ` (${ciText(score.ci)})` : ''}, clean-log cost ${pct(score.cost)}` }));
      g.append(dot);
      svg.append(g);
      marks.push({ x0: cx - 10, y0: cy - 10, x1: cx + 10, y1: cy + 10, w: 2 });
      pointLabels.push({ cls: `pt d-${def.id}`, cx, cy, text: def.name });
    }

    // Labels are drawn last, each at the nearby spot that least covers other labels, dots and whiskers (in that order of weight).
    const measure = (t) => {
      const b = t.getBBox();
      if (b.width) return { x0: b.x - 3, y0: b.y - 1, x1: b.x + b.width + 3, y1: b.y + b.height + 1, w: 4 };
      const n = t.textContent.length * 6.8, tx = Number(t.getAttribute('x')), ty = Number(t.getAttribute('y'));
      const a = t.getAttribute('text-anchor');
      const x0 = a === 'end' ? tx - n : a === 'middle' ? tx - n / 2 : tx;
      return { x0, y0: ty - 12, x1: x0 + n, y1: ty + 3, w: 4 };
    };
    const overlap = (a, b) => Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) * Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
    const area = { x0: m.l + 2, y0: 4, x1: W - 4, y1: H - m.b + 9 };
    const taken = [...marks, measure(better)];
    const SPOTS = [
      [12, 4, 'start'], [-12, 4, 'end'], [10, -12, 'start'], [10, 20, 'start'], [-10, -12, 'end'], [-10, 20, 'end'],
      [0, -15, 'middle'], [0, 25, 'middle'], [12, -24, 'start'], [12, 32, 'start'], [-12, -24, 'end'], [-12, 32, 'end'],
      [0, -30, 'middle'], [0, 40, 'middle'], [14, -38, 'start'], [-14, -38, 'end'], [14, 46, 'start'], [-14, 46, 'end'],
    ];
    const place = (host, cls, cx, cy, text, spots) => {
      const t = sv('text', { class: cls, text });
      host.append(t);
      let best = null;
      spots.forEach(([dx, dy, anchor], i) => {
        setAttrs(t, { x: (cx + dx).toFixed(1), y: (cy + dy).toFixed(1), 'text-anchor': anchor });
        const b = measure(t);
        const outside = (b.x0 < area.x0 || b.x1 > area.x1 || b.y0 < area.y0 || b.y1 > area.y1) ? 1e6 : 0;
        const cost = outside + taken.reduce((s, o) => s + o.w * overlap(b, o), 0) + i * 4;
        if (!best || cost < best.cost) best = { cost, dx, dy, anchor, b };
      });
      setAttrs(t, { x: (cx + best.dx).toFixed(1), y: (cy + best.dy).toFixed(1), 'text-anchor': best.anchor });
      taken.push(best.b);
      return best.b;
    };
    for (const a of anchorLabels) {
      const first = a.top ? [[-12, 18, 'end'], [12, 18, 'start']] : [[-12, -12, 'end'], [12, -12, 'start']];
      place(a.g, null, a.cx, a.cy, a.text, [...first, ...SPOTS]);
    }
    for (const p of pointLabels) {
      const g = sv('g', { class: p.cls });
      svg.append(g);
      const b = place(g, null, p.cx, p.cy, p.text, SPOTS);
      // A label pushed away from its dot gets a leader, so a tight cluster still reads unambiguously.
      const px = Math.min(Math.max(p.cx, b.x0), b.x1), py = Math.min(Math.max(p.cy, b.y0), b.y1);
      const dist = Math.hypot(px - p.cx, py - p.cy);
      if (dist > 16) {
        const ux = (px - p.cx) / dist, uy = (py - p.cy) / dist;
        g.prepend(sv('line', { class: 'leader', x1: (p.cx + ux * 9).toFixed(1), y1: (p.cy + uy * 9).toFixed(1), x2: px.toFixed(1), y2: py.toFixed(1) }));
      }
    }

    const b = benchById(state.bench);
    const following = $('#frontier-following');
    clear(following);
    append(following, ['Following the ', el('a', { href: '#scoreboard', text: 'scoreboard' }), ` selection: ${b.name} · ${taskById(state.task).name} · ${modelName()}.`]);
  }

  // ---------- §4 Transfer slope chart ----------

  function renderLegend() {
    toggleGroup($('#transfer-legend'), 'defences', defences().map((d) => ({
      key: d.id,
      attrs: { class: `d-${d.id}`, text: d.name, onclick: () => { state.focus = state.focus === d.id ? null : d.id; renderLegend(); applyFocus(); } },
    })), state.focus);
  }

  function applyFocus() {
    const svg = $('#bump-svg');
    svg.classList.toggle('has-focus', Boolean(state.focus));
    for (const g of svg.querySelectorAll('.bump-line')) {
      g.classList.toggle('focus', g.dataset.def === state.focus);
    }
  }

  function renderTransfer() {
    const svg = $('#bump-svg');
    for (const child of [...svg.childNodes]) {
      if (child.nodeName !== 'title' && child.nodeName !== 'desc') svg.removeChild(child);
    }
    const W = 760, H = 400, m = { t: 58, r: 70, b: 24, l: 236 };
    const benches = benchmarks().filter((b) => cell(b.id, state.task));
    const nDef = defences().length;
    const colX = (i) => (benches.length < 2 ? (m.l + W - m.r) / 2 : m.l + (i * (W - m.l - m.r)) / (benches.length - 1));
    const rowY = (rank) => m.t + ((rank - 1) * (H - m.t - m.b)) / Math.max(1, nDef - 1);
    // Tied defences share a rank, so their end labels are stacked instead of drawn on top of each other.
    const used = {};
    const labelY = (side, bi, rank) => {
      const slot = (used[`${side}|${bi}|${rank}`] = (used[`${side}|${bi}|${rank}`] ?? -1) + 1);
      return rowY(rank) + 4 + slot * 14;
    };

    if (isIllustrative()) {
      svg.append(sv('text', { class: 'watermark', x: W / 2, y: H / 2 + 20, 'text-anchor': 'middle', text: 'ILLUSTRATIVE' }));
    }

    const ranks = {};
    benches.forEach((b, bi) => {
      for (const r of ranked(b.id, state.task)) {
        if (r.rank) (ranks[r.def.id] ||= [])[bi] = { rank: r.rank, asr: r.score.asr };
      }
      const col = sv('g', { class: 'bump-col' });
      col.append(sv('line', { x1: colX(bi), x2: colX(bi), y1: m.t - 10, y2: H - m.b + 6 }));
      col.append(sv('text', { x: colX(bi), y: 20, 'text-anchor': 'middle', text: `${b.short} · ${b.name}` }));
      col.append(sv('text', { class: 'sub', x: colX(bi), y: 36, 'text-anchor': 'middle', text: b.family }));
      svg.append(col);
    });

    for (const d of defences()) {
      const pts = ranks[d.id];
      if (!pts) continue;
      const g = sv('g', { class: `bump-line d-${d.id}`, 'data-def': d.id });
      const coords = benches.map((_, bi) => pts[bi] && [colX(bi), rowY(pts[bi].rank)]).filter(Boolean);
      let dPath = '';
      coords.forEach(([cx, cy], i) => {
        if (i === 0) { dPath += `M${cx},${cy}`; return; }
        const [px, py] = coords[i - 1];
        const mx = (px + cx) / 2;
        dPath += ` C${mx},${py} ${mx},${cy} ${cx},${cy}`;
      });
      g.append(sv('path', { d: dPath }));
      benches.forEach((b, bi) => {
        const p = pts[bi];
        if (!p) return;
        const c = sv('circle', { cx: colX(bi), cy: rowY(p.rank), r: 9 });
        c.append(sv('title', { text: `${d.name} on ${b.name}: rank ${p.rank}, attack success ${pct(p.asr)}` }));
        g.append(c, sv('text', { class: 'rk', x: colX(bi), y: rowY(p.rank) + 3.5, 'text-anchor': 'middle', text: String(p.rank) }));
      });
      const first = pts.findIndex(Boolean);
      const last = pts.length - 1;
      if (first >= 0) {
        g.append(sv('text', { x: colX(first) - 16, y: labelY('l', first, pts[first].rank), 'text-anchor': 'end' },
          d.name, sv('tspan', { dx: 8, text: pct(pts[first].asr) })));
      }
      if (pts[last]) g.append(sv('text', { x: colX(last) + 16, y: labelY('r', last, pts[last].rank), text: pct(pts[last].asr) }));
      g.addEventListener('mouseenter', () => { if (!state.focus) { svg.classList.add('has-focus'); g.classList.add('focus'); } });
      g.addEventListener('mouseleave', () => { if (!state.focus) { svg.classList.remove('has-focus'); g.classList.remove('focus'); } });
      svg.append(g);
    }
    applyFocus();

    const following = $('#transfer-following');
    clear(following);
    append(following, ['Following the ', el('a', { href: '#scoreboard', text: 'scoreboard' }), ` selection: ${taskById(state.task).name} · ${modelName()}. Rank 1 = lowest attack success; exact ties share a rank.`]);
  }

  // ---------- Section highlight in the masthead ----------

  function wireToc() {
    const links = [...document.querySelectorAll('.toc a')];
    if (!('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        for (const a of links) a.classList.toggle('active', a.getAttribute('href') === `#${e.target.id}`);
      }
    }, { rootMargin: '-40% 0px -55% 0px' });
    for (const a of links) {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) io.observe(target);
    }
  }

  // ---------- Boot ----------

  function renderResults() {
    renderControls();
    renderGradient();
    renderBoard();
    renderFrontier();
    renderTransfer();
  }

  async function loadJson(path) {
    let res;
    try {
      res = await fetch(path, { cache: 'no-cache' });
    } catch {
      throw new DataError('fetch', `${path} could not be fetched`);
    }
    if (!res.ok) throw new DataError('fetch', `${path}: HTTP ${res.status}`);
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      throw new DataError('invalid', `${path} is not valid JSON (NaN and Infinity are not allowed)`);
    }
  }

  const usableCase = (c) => c && typeof c === 'object' && Array.isArray(c.lines) && c.lines.every(Array.isArray)
    && c.outcomes && typeof c.outcomes === 'object';

  async function boot() {
    wireToc();
    if (location.protocol === 'file:') {
      showRibbon('error', 'Data not loaded',
        'This page was opened from disk, where browsers block its data files. Serve this folder over HTTP '
        + '(python -m http.server 8000 --directory code/docs) and open http://localhost:8000/.');
      return;
    }

    const examplesLoad = loadJson('data/examples.json').catch(() => null);
    let flagged;
    try {
      const board = await loadJson('data/scoreboard.json');
      checkBoard(board);
      flagged = normaliseCells(board);
      state.board = board;
    } catch (err) {
      if (err instanceof DataError && err.kind === 'fetch') {
        showRibbon('error', 'Data not loaded', `${err.message}. Check that data/scoreboard.json sits next to index.html.`);
      } else {
        showRibbon('error', 'Data invalid',
          `${err instanceof DataError ? err.message : 'data/scoreboard.json has an unexpected shape'}. Regenerate it with build_dashboard.py.`);
      }
      return;
    }
    const board = state.board;
    state.bench = board.benchmarks[0].id;
    state.task = board.tasks[0].id;
    state.defence = board.defences[0].id;

    const examples = await examplesLoad;
    const cases = Array.isArray(examples?.cases) ? examples.cases.filter(usableCase) : [];
    state.examples = { ...(examples || {}), cases };

    // Only the scoreboard data decides whether the numbers are placeholders. The examples are
    // hand-written by design, so they are labelled where they are shown, not page-wide.
    if (isIllustrative()) {
      showRibbon('info', 'Work in progress',
        'Every number on this page is an illustrative placeholder. No experiment has been run yet.');
      $('#sticker').hidden = false;
    }
    const missing = Array.isArray(board.models_missing) ? board.models_missing.map(String) : [];
    if (board.partial === true || missing.length) {
      showRibbon('info', 'Partial results', missing.length
        ? `Not every model has finished: ${missing.join(', ')} ${missing.length === 1 ? 'is' : 'are'} not in these numbers yet. Averages cover only the models shown.`
        : 'Not every model has finished. Averages cover only the models shown.');
    }
    if (flagged.length) {
      showRibbon('info', 'Data warning',
        `Some values in data/scoreboard.json are malformed and are shown as a dash (${flagged.join(', ')}).`);
    }
    if (typeof board.status_note === 'string' && board.status_note) {
      const status = $('#status-note');
      status.textContent = `Status: ${board.status_note}`;
      status.hidden = false;
    }

    const note = $('#examples-note');
    if (!examples || !cases.length) {
      note.textContent = 'The example attacks could not be loaded from data/examples.json, so this plate is empty.';
      note.hidden = false;
    } else if (state.examples.illustrative) {
      note.textContent = 'Hand-written examples with synthetic log lines. The responses show how each defence '
        + 'behaves; they are not model output. Measured results are in Plate 02.';
      note.hidden = false;
    }

    try {
      renderTally();
      if (cases.length) { renderFragment(); renderAnatomy(); }
      renderLegend();
      renderResults();
    } catch {
      showRibbon('error', 'Data invalid', 'data/scoreboard.json could not be drawn. Regenerate it with build_dashboard.py.');
    }
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
