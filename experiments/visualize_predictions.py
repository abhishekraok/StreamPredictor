"""
Self-contained HTML report visualizing byte-level predictions of a trained
StreamPredictor on held-out PTB bytes.

Every held-out position is traced through Generator.next_word_distribution_traced:
every matching-suffix Pop, its blend weight, its top votes, and how much of
the actual byte's probability came from each Pop vs the uniform floor. The
report shows 20 random + 20 worst-loss positions in full detail, plus a
clickable per-byte loss ribbon over the whole test slice and aggregates
(bits histogram, loss vs longest-match length, worst-offender Pops).

Reuses PatternStore/ptb_bpc_byte.tsv when present (the model saved by
experiments/bpc_byte.py); delete that file to retrain from scratch.
Outputs PatternStore/prediction_report.html and prediction_trace.jsonl.
"""
import gzip
import json
import random
from pathlib import Path

import numpy as np

from streampredictor.stream_predictor import StreamPredictor
from streampredictor import data_fetcher, generator as generator_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
input_text_file = REPO_ROOT / 'data' / 'ptb.test.txt'
pattern_store = REPO_ROOT / 'PatternStore'
pattern_store.mkdir(exist_ok=True)
model_file = pattern_store / 'ptb_bpc_byte.tsv'
report_file = pattern_store / 'prediction_report.html'
trace_file = pattern_store / 'prediction_trace.jsonl'

# Same data/config as experiments/bpc_byte.py, fixed to the tuned variant.
max_bytes = 2 * 10 ** 5
test_length = -10 ** 4
uniform_ratio = 1e-3
config_label = "strength_blend + log_strength, uniform_ratio={0}".format(uniform_ratio)

n_random_samples = 20
n_worst_samples = 20
top_k_final = 10
top_k_votes = 5
context_before = 60
context_after = 12
rng_seed = 42
fallback_threshold = 0.5  # uniform_share above this counts as "model knew nothing"


def get_model_and_tokens():
    tokens = data_fetcher.get_bytes_from_file(str(input_text_file), max_bytes)
    train_tokens, test_tokens = tokens[:test_length], tokens[test_length:]
    print('Read {0} bytes; train {1}, test {2}'.format(
        len(tokens), len(train_tokens), len(test_tokens)))
    sp = StreamPredictor()
    loaded = False
    if model_file.exists():
        # Byte-level patterns can contain raw \n and \t bytes, which the
        # line-based TSV format does not escape; fall back to retraining
        # when the saved model does not round-trip.
        try:
            sp.file_manager.load_tsv(str(model_file))
            sp.pop_manager.add_words_to_vocabulary(train_tokens, verbose=False)
            loaded = True
            print('Reusing trained model {0} (delete it to retrain)'.format(model_file))
        except (IndexError, KeyError, ValueError) as e:
            print('Could not load {0} ({1!r}); retraining'.format(model_file, e))
            sp = StreamPredictor()
    if not loaded:
        sp.train(train_tokens)
        sp.file_manager.save_tsv(str(model_file))
    sp.generator = generator_mod.Generator(
        sp.pop_manager.pattern_collection,
        sp.pop_manager.vocabulary,
        prediction_mode='strength_blend',
        blend_log_strength=True,
    )
    sp.generator.uniform_prediction_ratio = uniform_ratio
    return sp, test_tokens


def trace_positions(sp, test_tokens):
    """One record per scored position i (test_tokens[i], i >= 1)."""
    sp.pop_manager.add_words_to_vocabulary(test_tokens, verbose=False)
    gen = sp.generator
    records = []
    n = len(test_tokens)
    for i in range(1, n):
        final, components = gen.next_word_distribution_traced(test_tokens[:i])
        actual = test_tokens[i]
        z = sum(w * sum(d.values()) for _, w, d in components)
        total_weight = sum(w for _, w, _ in components)
        p_actual = final.get(actual, 0.0)
        bits = float(-np.log2(max(p_actual, 1e-300)))
        uniform_contribution = 0.0
        pops = []
        for source, w, d in components:
            contribution = float(w * d.get(actual, 0.0) / z)
            if source is None:
                uniform_contribution = contribution
                continue
            votes = sorted(d.items(), key=lambda kv: -kv[1])[:top_k_votes]
            pops.append({
                'pattern': source.unrolled_pattern,
                'length': len(source.unrolled_pattern),
                'strength': source.strength,
                'split': (source.first_component.unrolled_pattern + '\x00' +
                          source.second_component.unrolled_pattern)
                         if source.first_component and source.second_component else None,
                'weight': float(w),
                'weight_share': float(w / total_weight) if total_weight > 0 else 0.0,
                'p_in_pop': float(d.get(actual, 0.0)),
                'contribution': contribution,
                'votes': [[t, float(p)] for t, p in votes],
            })
        pops.sort(key=lambda p: -p['weight_share'])
        ranked = sorted(final.items(), key=lambda kv: -kv[1])
        final_top = [[t, float(p)] for t, p in ranked[:top_k_final]]
        actual_rank = next(r for r, (t, _) in enumerate(ranked) if t == actual) + 1
        records.append({
            'pos': i,
            'actual': actual,
            'p_actual': float(p_actual),
            'bits': bits,
            'uniform_share': float(uniform_contribution / p_actual) if p_actual > 0 else 1.0,
            'longest': max((p['length'] for p in pops), default=0),
            'n_pops': len(pops),
            'final_top': final_top,
            'actual_rank': actual_rank,
            'pops': pops,
        })
        if i % 1000 == 0:
            running_bpc = sum(r['bits'] for r in records) / len(records)
            print('Traced {0}/{1} positions, running bpc {2:.4f}'.format(i, n - 1, running_bpc))
    return records


def select_samples(records):
    worst = sorted(range(len(records)), key=lambda k: -records[k]['bits'])[:n_worst_samples]
    worst_set = set(worst)
    rest = [k for k in range(len(records)) if k not in worst_set]
    rng = random.Random(rng_seed)
    random_samples = sorted(rng.sample(rest, min(n_random_samples, len(rest))))
    return random_samples, worst


def aggregate(records):
    total_bits = sum(r['bits'] for r in records)
    hist_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8+']
    hist = [0] * len(hist_labels)
    for r in records:
        hist[min(int(r['bits']), 8)] += 1

    def len_bin(length):
        if length <= 5:
            return str(length)
        if length <= 9:
            return '6-9'
        if length <= 19:
            return '10-19'
        return '20+'

    len_order = ['0', '1', '2', '3', '4', '5', '6-9', '10-19', '20+']
    len_stats = {}
    for r in records:
        b = len_bin(r['longest'])
        cnt, bits = len_stats.get(b, (0, 0.0))
        len_stats[b] = (cnt + 1, bits + r['bits'])
    match_len = [[b, len_stats[b][0], len_stats[b][1] / len_stats[b][0]]
                 for b in len_order if b in len_stats]

    offenders = {}
    for r in records:
        for p in r['pops']:
            blame, cnt, share, p_in = offenders.get(p['pattern'], (0.0, 0, 0.0, 0.0))
            offenders[p['pattern']] = (blame + p['weight_share'] * r['bits'],
                                       cnt + 1,
                                       share + p['weight_share'],
                                       p_in + p['p_in_pop'])
    top_offenders = sorted(offenders.items(), key=lambda kv: -kv[1][0])[:20]
    offender_rows = [[pat, blame, cnt, share / cnt, p_in / cnt]
                     for pat, (blame, cnt, share, p_in) in top_offenders]

    fallback = [r for r in records if r['uniform_share'] > fallback_threshold]
    return {
        'total_bits': total_bits,
        'bpc': total_bits / len(records),
        'hist': list(zip(hist_labels, hist)),
        'match_len': match_len,
        'offenders': offender_rows,
        'fallback_rate': len(fallback) / len(records),
        'fallback_bits_share': sum(r['bits'] for r in fallback) / total_bits if total_bits else 0.0,
    }


def build_card(record, test_tokens):
    i = record['pos']
    actual_set = {record['actual']}
    return {
        'pos': i,
        'actual': record['actual'],
        'bits': round(record['bits'], 3),
        'p_actual': record['p_actual'],
        'uniform_share': round(record['uniform_share'], 4),
        'actual_rank': record['actual_rank'],
        'before': ''.join(test_tokens[max(0, i - context_before):i]),
        'after': ''.join(test_tokens[i + 1:i + 1 + context_after]),
        'final_top': [[t, round(p, 6), t in actual_set] for t, p in record['final_top']],
        'pops': [{
            'pattern': p['pattern'],
            'split': p['split'],
            'length': p['length'],
            'strength': p['strength'],
            'weight': round(p['weight'], 4),
            'weight_share': round(p['weight_share'], 5),
            'p_in_pop': round(p['p_in_pop'], 5),
            'contribution': round(p['contribution'], 6),
            'votes': [[t, round(v, 4)] for t, v in p['votes']],
        } for p in record['pops']],
    }


def build_data(records, test_tokens, random_samples, worst_samples, agg, gzip_bpc, pattern_count):
    ribbon = [[r['actual'], round(r['bits'], 3), round(r['uniform_share'], 3),
               r['n_pops'], r['longest']] for r in records]
    sample_keys = set(random_samples) | set(worst_samples)
    cards = {str(k): build_card(records[k], test_tokens) for k in sample_keys}
    return {
        'meta': {
            'config': config_label,
            'positions': len(records),
            'bpc': round(agg['bpc'], 4),
            'gzip_bpc': round(gzip_bpc, 4),
            'pattern_count': pattern_count,
            'fallback_rate': round(agg['fallback_rate'], 4),
            'fallback_bits_share': round(agg['fallback_bits_share'], 4),
            'fallback_threshold': fallback_threshold,
        },
        'ribbon': ribbon,
        'sample_groups': {'random': random_samples, 'worst': worst_samples},
        'cards': cards,
        'hist': agg['hist'],
        'match_len': [[b, c, round(m, 3)] for b, c, m in agg['match_len']],
        'offenders': [[pat, round(blame, 2), cnt, round(share, 4), round(p_in, 4)]
                      for pat, blame, cnt, share, p_in in agg['offenders']],
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>StreamPredictor prediction report</title>
<style>
  body { background: #14161a; color: #d8dce2; font-family: 'DejaVu Sans Mono', Consolas, monospace;
         font-size: 13px; margin: 0; padding: 24px 32px; }
  h1 { font-size: 18px; } h2 { font-size: 15px; margin-top: 36px; border-bottom: 1px solid #333a44;
         padding-bottom: 4px; } h3 { font-size: 13px; }
  .meta { color: #9aa3af; margin-bottom: 4px; }
  .statrow { display: flex; gap: 28px; flex-wrap: wrap; margin: 14px 0; }
  .stat { background: #1c2027; border: 1px solid #2a313c; border-radius: 6px; padding: 8px 14px; }
  .stat .v { font-size: 17px; color: #fff; } .stat .l { color: #8b95a3; font-size: 11px; }
  #ribbon { line-height: 1.7; word-break: break-all; background: #0e1013; padding: 12px;
            border-radius: 6px; max-height: 420px; overflow-y: auto; }
  .rb { color: #fff; padding: 0 1px; cursor: default; }
  .rb.sampled { outline: 2px solid #ffd23f; cursor: pointer; }
  #tip { position: fixed; background: #262c36; border: 1px solid #4a5462; padding: 6px 9px;
         border-radius: 5px; pointer-events: none; display: none; z-index: 10; font-size: 12px;
         white-space: pre; }
  .legend span { padding: 1px 7px; margin-right: 3px; color: #fff; }
  .card { background: #1a1e24; border: 1px solid #2a313c; border-radius: 8px;
          padding: 14px 18px; margin: 14px 0; }
  .card.flash { outline: 2px solid #ffd23f; }
  .card h3 { margin: 0 0 8px 0; color: #ffd23f; }
  .context { background: #0e1013; padding: 8px 10px; border-radius: 5px; margin: 8px 0;
             word-break: break-all; }
  .context .tgt { background: #c0392b; color: #fff; padding: 0 2px; font-weight: bold; }
  .context .ctx { color: #7f8a98; }
  table { border-collapse: collapse; margin: 8px 0; width: 100%; }
  th, td { border: 1px solid #2a313c; padding: 3px 8px; text-align: left; font-size: 12px; }
  th { background: #20252d; color: #9aa3af; font-weight: normal; }
  td.num { text-align: right; }
  .bar { background: #3a7bd5; height: 11px; display: inline-block; vertical-align: middle; }
  .actualrow { background: #243321; } .actualrow td { color: #b8e6a0; }
  .pat { color: #e8c177; } .voteactual { color: #b8e6a0; font-weight: bold; }
  .dim { color: #6b7585; }
  a { color: #6fb3ff; text-decoration: none; cursor: pointer; }
  .twocol { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-start; }
  .twocol > div { flex: 1; min-width: 340px; }
</style>
</head>
<body>
<h1>StreamPredictor byte-level prediction report</h1>
<div class="meta" id="metaline"></div>
<div class="statrow" id="stats"></div>

<h2>Loss ribbon <span class="dim">(each test byte colored by its bits; yellow outline = sampled below, click to open)</span></h2>
<div class="legend" id="legend"></div>
<div id="ribbon"></div>

<h2>Worst-loss samples</h2>
<div id="worst"></div>

<h2>Random samples</h2>
<div id="random"></div>

<h2>Aggregates</h2>
<div class="twocol">
  <div><h3>Bits per position</h3><table id="hist"></table></div>
  <div><h3>Mean bits vs longest-match length</h3><table id="matchlen"></table></div>
</div>
<h3>Worst-offender Pops <span class="dim">(blame = sum over positions of weight_share x bits)</span></h3>
<table id="offenders"></table>

<div id="tip"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const VIS = {' ': '\\u2423', '\\n': '\\u21b5', '\\t': '\\u21e5'};
function vis(s) { return s.split('').map(c => VIS[c] || c).join(''); }
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function ev(s) { return esc(vis(s)); }
function pct(x) { return (100 * x).toFixed(x >= 0.095 ? 1 : 2) + '%'; }
function bitsColor(b) {
  if (b > 8) return '#8b2fc9';
  const hue = 120 * (1 - Math.min(b, 8) / 8);
  return 'hsl(' + hue + ', 70%, 34%)';
}
function trunc(s, n) { return s.length > n ? s.slice(0, n) + '\\u2026' : s; }

// header
const m = DATA.meta;
document.getElementById('metaline').textContent =
  m.config + ' | ' + m.positions + ' scored positions | ' + m.pattern_count + ' patterns in collection';
function stat(v, l) { return '<div class="stat"><div class="v">' + v + '</div><div class="l">' + l + '</div></div>'; }
document.getElementById('stats').innerHTML =
  stat(m.bpc, 'model bpc (intrinsic)') +
  stat(m.gzip_bpc, 'gzip bpc, same bytes') +
  stat(pct(m.fallback_rate), 'positions surviving mostly on uniform floor (&gt;' + (100 * m.fallback_threshold) + '% of p)') +
  stat(pct(m.fallback_bits_share), 'share of total bits from those positions');

// legend
let lg = '';
for (let b = 0; b <= 8; b += 2) lg += '<span style="background:' + bitsColor(b + 0.01) + '">' + b + ' bits</span>';
lg += '<span style="background:#8b2fc9">8+</span>';
document.getElementById('legend').innerHTML = lg;

// ribbon
const sampledSet = new Set([].concat(DATA.sample_groups.worst, DATA.sample_groups.random));
let rb = '';
DATA.ribbon.forEach((r, k) => {
  const cls = sampledSet.has(k) ? 'rb sampled' : 'rb';
  rb += '<span class="' + cls + '" data-k="' + k + '" style="background:' + bitsColor(r[1]) + '">' + ev(r[0]) + '</span>';
  if (r[0] === '\\n') rb += '<br>';
});
const ribbonEl = document.getElementById('ribbon');
ribbonEl.innerHTML = rb;

const tip = document.getElementById('tip');
ribbonEl.addEventListener('mousemove', e => {
  const t = e.target.closest('.rb');
  if (!t) { tip.style.display = 'none'; return; }
  const r = DATA.ribbon[+t.dataset.k];
  tip.textContent = 'pos ' + (+t.dataset.k + 1) + '  byte ' + JSON.stringify(r[0]) +
    '\\nbits ' + r[1] + '  uniform share ' + pct(r[2]) +
    '\\nmatching pops ' + r[3] + '  longest match ' + r[4];
  tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 260) + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
});
ribbonEl.addEventListener('mouseleave', () => tip.style.display = 'none');
ribbonEl.addEventListener('click', e => {
  const t = e.target.closest('.rb.sampled');
  if (!t) return;
  const card = document.getElementById('card-' + t.dataset.k);
  card.scrollIntoView({behavior: 'smooth', block: 'center'});
  card.classList.add('flash');
  setTimeout(() => card.classList.remove('flash'), 1600);
});

// cards
function renderCard(k) {
  const c = DATA.cards[String(k)];
  let h = '<div class="card" id="card-' + k + '">';
  h += '<h3>position ' + c.pos + ' &mdash; actual byte ' + esc(JSON.stringify(c.actual)) +
       ' &mdash; ' + c.bits + ' bits (p=' + c.p_actual.toExponential(2) +
       ', rank ' + c.actual_rank + ', uniform share ' + pct(c.uniform_share) + ') ' +
       '<a data-jump="' + k + '">show in ribbon</a></h3>';
  h += '<div class="context"><span class="ctx">' + ev(c.before) + '</span>' +
       '<span class="tgt">' + ev(c.actual) + '</span>' +
       '<span class="ctx">' + ev(c.after) + '</span></div>';

  h += '<table><tr><th>final top-' + c.final_top.length + '</th><th>p</th><th></th></tr>';
  const maxp = c.final_top.length ? c.final_top[0][1] : 1;
  c.final_top.forEach(row => {
    h += '<tr' + (row[2] ? ' class="actualrow"' : '') + '><td>' + ev(row[0]) +
         '</td><td class="num">' + pct(row[1]) +
         '</td><td><span class="bar" style="width:' + Math.max(1, 160 * row[1] / maxp) + 'px"></span></td></tr>';
  });
  if (!c.final_top.some(r => r[2])) {
    h += '<tr class="actualrow"><td>' + ev(c.actual) + '</td><td class="num">' + pct(c.p_actual) +
         '</td><td class="dim">rank ' + c.actual_rank + ', below top-' + c.final_top.length + '</td></tr>';
  }
  h += '</table>';

  h += '<table><tr><th>matching Pop (suffix, longest first by weight)</th><th>len</th><th>strength</th>' +
       '<th>weight share</th><th>p(actual | pop)</th><th>contrib to p(actual)</th><th>top votes</th></tr>';
  c.pops.forEach(p => {
    const split = p.split ? ' <span class="dim">= ' + ev(trunc(p.split.split('\\u0000')[0], 20)) +
        '\\u00b7' + ev(trunc(p.split.split('\\u0000')[1], 20)) + '</span>' : '';
    const votes = p.votes.map(v =>
        '<span class="' + (v[0] === c.actual ? 'voteactual' : '') + '">' + ev(v[0]) + ':' + pct(v[1]) + '</span>'
      ).join(' ');
    h += '<tr><td><span class="pat" title="' + esc(p.pattern) + '">' + ev(trunc(p.pattern, 44)) + '</span>' + split +
         '</td><td class="num">' + p.length + '</td><td class="num">' + p.strength +
         '</td><td class="num">' + pct(p.weight_share) + '</td><td class="num">' + pct(p.p_in_pop) +
         '</td><td class="num">' + pct(p.contribution) + '</td><td>' + votes + '</td></tr>';
  });
  h += '</table></div>';
  return h;
}
document.getElementById('worst').innerHTML = DATA.sample_groups.worst.map(renderCard).join('');
document.getElementById('random').innerHTML = DATA.sample_groups.random.map(renderCard).join('');
document.body.addEventListener('click', e => {
  const a = e.target.closest('a[data-jump]');
  if (!a) return;
  const span = ribbonEl.querySelector('.rb[data-k="' + a.dataset.jump + '"]');
  span.scrollIntoView({block: 'center'});
});

// aggregates
function barTable(el, rows, header, maxv) {
  let h = '<tr>' + header.map(x => '<th>' + x + '</th>').join('') + '<th></th></tr>';
  rows.forEach(r => {
    h += '<tr>' + r.map((x, i) => '<td class="' + (i ? 'num' : '') + '">' + x + '</td>').join('') +
         '<td><span class="bar" style="width:' + Math.max(1, 180 * r[1] / maxv) + 'px"></span></td></tr>';
  });
  el.innerHTML = h;
}
barTable(document.getElementById('hist'), DATA.hist, ['bits', 'positions'],
         Math.max(...DATA.hist.map(r => r[1])));
barTable(document.getElementById('matchlen'), DATA.match_len, ['longest match len', 'positions', 'mean bits'],
         Math.max(...DATA.match_len.map(r => r[1])));
let oh = '<tr><th>pattern</th><th>blame (bits)</th><th>positions</th><th>avg weight share</th><th>avg p(actual | pop)</th></tr>';
DATA.offenders.forEach(r => {
  oh += '<tr><td><span class="pat" title="' + esc(r[0]) + '">' + ev(trunc(r[0], 60)) + '</span></td>' +
        '<td class="num">' + r[1] + '</td><td class="num">' + r[2] + '</td>' +
        '<td class="num">' + pct(r[3]) + '</td><td class="num">' + pct(r[4]) + '</td></tr>';
});
document.getElementById('offenders').innerHTML = oh;
</script>
</body>
</html>
"""


def write_report(data):
    payload = json.dumps(data, ensure_ascii=True).replace('</', '<\\/')
    report_file.write_text(HTML_TEMPLATE.replace('__DATA__', payload), encoding='utf-8')
    print('Wrote report ' + str(report_file))


def write_trace(records):
    with open(trace_file, 'w') as f:
        for r in records:
            f.write(json.dumps({
                'pos': r['pos'], 'actual': r['actual'], 'p_actual': r['p_actual'],
                'bits': round(r['bits'], 4), 'uniform_share': round(r['uniform_share'], 4),
                'n_pops': r['n_pops'], 'longest': r['longest'],
                'actual_rank': r['actual_rank'],
            }, ensure_ascii=True) + '\n')
    print('Wrote trace ' + str(trace_file))


if __name__ == '__main__':
    sp, test_tokens = get_model_and_tokens()
    records = trace_positions(sp, test_tokens)
    random_samples, worst_samples = select_samples(records)
    agg = aggregate(records)
    scored_text = ''.join(test_tokens[1:]).encode('latin-1')
    gzip_bpc = len(gzip.compress(scored_text)) * 8 / len(scored_text)
    data = build_data(records, test_tokens, random_samples, worst_samples, agg,
                      gzip_bpc, len(sp.pop_manager.pattern_collection))
    write_report(data)
    write_trace(records)
    print('bpc {0:.4f} (gzip {1:.4f}); fallback rate {2:.1%} carrying {3:.1%} of bits'.format(
        agg['bpc'], gzip_bpc, agg['fallback_rate'], agg['fallback_bits_share']))
    print('Open: file://' + str(report_file))
