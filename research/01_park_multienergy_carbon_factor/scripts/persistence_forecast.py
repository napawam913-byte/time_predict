#!/usr/bin/env python3
"""CPU-only electricity Persistence; fixed hourly, release-aware backtesting."""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import sys

HOUR = timedelta(hours=1)
UTC = timezone.utc
TARGET = 'electricity_factor_kgco2_per_kwh'
INPUT_SHA256 = 'ca206f709fe33dffe9785e01656927c8dcfd07dcba541250d35026645fa4a200'
RESEARCH = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = RESEARCH / 'runs/park_e_g_h_v0_1_20261008/history_hourly.csv'
BOUNDS = {
    'train': (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 9, 1, tzinfo=UTC)),
    'validation': (datetime(2023, 9, 1, tzinfo=UTC), datetime(2023, 11, 1, tzinfo=UTC)),
    'test': (datetime(2023, 11, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
}


@dataclass(frozen=True)
class Observation:
    start: datetime
    end: datetime
    split: str
    available: datetime
    value: float | None


def iso(value):
    return value.isoformat().replace('+00:00', 'Z') if value is not None else None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError('Timestamps must explicitly use UTC')
    if result.minute or result.second or result.microsecond:
        raise ValueError('Timestamps must lie on whole UTC hours')
    return result


def read_history(path):
    """Validate the frozen hourly schema; never repair, sort or fill input."""
    required = {'period_start', 'period_end', 'split', TARGET,
                'electricity_factor_valid', 'electricity_available_at'}
    result = []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if not required.issubset(names) or len(names) != len(set(names)):
            raise ValueError('Missing or duplicate CSV column names')
        for number, raw in enumerate(reader, start=2):
            if None in raw or any(raw.get(k) is None for k in required):
                raise ValueError(f'Malformed CSV row {number}')
            start, end = timestamp(raw['period_start']), timestamp(raw['period_end'])
            available = timestamp(raw['electricity_available_at'])
            split = raw['split']
            if end != start + HOUR or (result and start != result[-1].end):
                raise ValueError(f'Non-contiguous or non-hourly grid at row {number}')
            if available != end + HOUR:
                raise ValueError(f'Expected simulated one-hour release delay at row {number}')
            if split not in BOUNDS or not BOUNDS[split][0] <= start < BOUNDS[split][1]:
                raise ValueError(f'Unexpected time split at row {number}')
            flag = raw['electricity_factor_valid']
            if flag not in ('True', 'False'):
                raise ValueError(f'Invalid boolean mask at row {number}')
            value = None if raw[TARGET] == '' else float(raw[TARGET])
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f'Invalid factor at row {number}')
            if (value is not None) != (flag == 'True'):
                raise ValueError(f'Value/mask mismatch at row {number}')
            result.append(Observation(start, end, split, available, value))
    if not result:
        raise ValueError('Empty history')
    return result


def forecast(data, split, lookback=168, horizon=24):
    """Keep every target position; information and scoring masks are separate."""
    if split not in BOUNDS or lookback < 2 or horizon < 1:
        raise ValueError('Invalid split, lookback or horizon')
    audit = {'eligible_origins': 0, 'excluded_warmup_origins': 0,
             'excluded_target_boundary_origins': 0, 'origins_without_available_input': 0,
             'origins_with_extra_staleness': 0, 'max_input_age_hours_since_period_end': None}
    predictions = []
    ages = []
    for i, current in enumerate(data):
        if current.split != split:
            continue
        if i < lookback:
            audit['excluded_warmup_origins'] += 1
            continue
        targets = data[i:i + horizon]
        if len(targets) != horizon or any(t.split != split for t in targets):
            audit['excluded_target_boundary_origins'] += 1
            continue
        origin = current.start
        source = next((r for r in reversed(data[i-lookback:i])
                       if r.value is not None and r.available <= origin), None)
        age = (origin - source.end) / HOUR if source else None
        audit['eligible_origins'] += 1
        if source is None:
            audit['origins_without_available_input'] += 1
        else:
            ages.append(age)
            audit['origins_with_extra_staleness'] += int(age > 1)
        for step, target in enumerate(targets, start=1):
            predictions.append({
                'model': 'Persistence', 'split': split, 'origin': iso(origin),
                'history_start': iso(data[i-lookback].start),
                'target_start': iso(target.start), 'target_end': iso(target.end),
                'horizon_step': step, 'y_true': target.value,
                'y_pred': source.value if source else None,
                'target_valid': target.value is not None,
                'prediction_valid': source is not None,
                'score_valid': source is not None and target.value is not None,
                'source_period_start': iso(source.start) if source else None,
                'source_period_end': iso(source.end) if source else None,
                'source_available_at': iso(source.available) if source else None,
                'input_age_hours_since_period_end': age,
            })
    audit['max_input_age_hours_since_period_end'] = max(ages, default=None)
    audit['output_positions'] = len(predictions)
    audit['missing_target_positions'] = sum(not p['target_valid'] for p in predictions)
    audit['prediction_coverage'] = (sum(p['prediction_valid'] for p in predictions) / len(predictions)
                                    if predictions else None)
    return predictions, audit


def score_predictions(predictions, horizon):
    """Score forecast-origin/step pairs, not unique-hour emission totals."""
    def score(points):
        errors = [p['y_pred'] - p['y_true'] for p in points if p['score_valid']]
        n = len(errors)
        mse = math.fsum(e * e for e in errors) / n if n else None
        return {'n': n, 'mae': math.fsum(abs(e) for e in errors) / n if n else None,
                'mse': mse, 'rmse': math.sqrt(mse) if n else None,
                'bias_pred_minus_true': math.fsum(errors) / n if n else None}
    return {'unit': 'kgCO2/kWh', 'mse_unit': '(kgCO2/kWh)^2',
            'overall': score(predictions),
            'by_horizon': {str(h): score(p for p in predictions if p['horizon_step'] == h)
                           for h in range(1, horizon + 1)}}


def write_csv(path, points):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_first_window_svg(path, points):
    """Portable static figure; missing values break lines, no synthetic bounds."""
    width, height = 1000, 460
    left, right, top, bottom = 90, 935, 110, 330
    values = [p[k] for p in points for k in ('y_true', 'y_pred') if p[k] is not None]
    ymax = max(max(values, default=0) * 1.15, 0.01)
    def x(i):
        return left + (right-left) * i / max(len(points)-1, 1)
    def y(v):
        return bottom - (bottom-top) * v / ymax
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
               '<title>Electricity carbon factor: first forecast window</title>',
               '<desc>First chronological forecast, not a representative performance claim. Reference is a public estimate; missing points are gaps. No uncertainty interval estimated.</desc>',
               '<rect width="100%" height="100%" fill="white"/>',
               '<g font-family="sans-serif" font-size="14" fill="#252525">',
               '<text x="90" y="30" font-size="20">Electricity carbon factor: first forecast window</text>',
               f'<text x="90" y="56">Origin: {escape(points[0]["origin"])} | unit: kgCO2/kWh</text>',
               '<line x1="90" y1="80" x2="125" y2="80" stroke="#2463a6" stroke-width="2"/>',
               '<text x="133" y="85">Reference (public estimate)</text>',
               '<line x1="390" y1="80" x2="425" y2="80" stroke="#444444" stroke-width="2" stroke-dasharray="7 5"/>',
               '<text x="433" y="85">Persistence</text>']
    for tick in range(5):
        value = ymax * tick / 4
        content += [f'<line x1="{left}" y1="{y(value):.2f}" x2="{right}" y2="{y(value):.2f}" stroke="#e5e7eb"/>',
                    f'<text x="80" y="{y(value)+5:.2f}" text-anchor="end">{value:.3f}</text>']
    for key, color, dashed in [('y_true', '#2463a6', ''), ('y_pred', '#444444', 'stroke-dasharray="7 5"')]:
        commands, continuous = [], False
        for i, point in enumerate(points):
            value = point[key]
            if value is None:
                continuous = False
                continue
            commands.append(f'{"L" if continuous else "M"} {x(i):.2f} {y(value):.2f}')
            continuous = True
        content.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="2" {dashed}/>')
        for i, point in enumerate(points):
            if point[key] is not None:
                content.append(f'<circle cx="{x(i):.2f}" cy="{y(point[key]):.2f}" r="2.5" fill="{color}"/>')
    content.append(f'<path d="M {left} {top} V {bottom} H {right}" fill="none" stroke="#666666"/>')
    for i in sorted({0, len(points)//2, len(points)-1}):
        stamp = points[i]['target_start']
        content += [f'<text x="{x(i):.2f}" y="352" text-anchor="middle">{stamp[5:10]} {stamp[11:16]}</text>']
    content += ['<text x="510" y="379" text-anchor="middle">Target interval start (UTC)</text>',
                '<text x="90" y="408">First chronological window only. Missing reference values remain gaps.</text>',
                '<text x="90" y="432">Source values and availability timestamps: first_window.csv; aggregate results: metrics.json.</text>',
                '</g></svg>']
    with path.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(content) + '\n')


def run(history, output, split='validation', lookback=168, horizon=24, expected_sha256=None):
    history, output = Path(history).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    input_sha = sha256(history)
    if expected_sha256 is not None and input_sha != expected_sha256:
        raise ValueError('Input SHA-256 differs from the frozen history; input was not modified')
    data = read_history(history)
    predictions, audit = forecast(data, split, lookback, horizon)
    metrics = score_predictions(predictions, horizon)
    if not predictions:
        raise ValueError('No complete forecast windows; no output directory was created')
    first = predictions[:horizon]
    config = {'model': 'Persistence', 'protocol': 'park_electricity_persistence_v1',
              'target': TARGET, 'split': split, 'lookback_hours': lookback,
              'horizon_hours': horizon, 'origin_stride_hours': 1,
              'history': str(history), 'input_sha256': input_sha,
              'script_sha256': sha256(__file__), 'python': sys.version,
              'release_delay_hours': 1, 'release_delay_is_simulated': True,
              'history_rule': '[origin-L hours, origin); require available_at <= origin',
              'missing_input_rule': 'latest valid available value inside history window; otherwise null',
              'missing_target_rule': 'preserve null; score_valid = target_valid AND prediction_valid',
              'aggregation': 'equal weight per origin/target-step pair; not unique-hour emission totals',
              'training_performed': False, 'standardization_performed': False,
              'created_at_utc': iso(datetime.now(UTC))}
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / 'config.json', config)
        write_json(output / 'audit.json', audit)
        write_json(output / 'metrics.json', metrics)
        write_csv(output / 'predictions.csv', predictions)
        write_csv(output / 'first_window.csv', first)
        write_first_window_svg(output / 'first_window.svg', first)
        with (output / 'script_snapshot.py').open('xb') as stream:
            stream.write(Path(__file__).read_bytes())
        score = metrics['overall']
        mae_text = f"{score['mae']:.9f}" if score['mae'] is not None else 'NA（无可评分元素）'
        rmse_text = f"{score['rmse']:.9f}" if score['rmse'] is not None else 'NA（无可评分元素）'
        summary = f'''# 电力因子 Persistence：{split}

仅CPU确定性预测，无训练、无标准化。L={lookback}小时，H={horizon}小时，逐小时滚动起点。
电因子来源为GB公开估计，1小时发布延迟是研究假设，不代表实时部署已验证。

- 有效起点：{audit['eligible_origins']}；输出位置：{len(predictions)}。
- 可评分位置：{score['n']}；缺失目标位置：{audit['missing_target_positions']}。
- MAE：{mae_text} kgCO₂/kWh；RMSE：{rmse_text} kgCO₂/kWh。
- 输入额外陈旧起点：{audit['origins_with_extra_staleness']}；来源区间结束距起点最大{audit['max_input_age_hours_since_period_end']}小时。

[预测明细](predictions.csv) · [指标](metrics.json) · [首窗明细](first_window.csv) · [首窗曲线](first_window.svg) · [协议快照](config.json) · [审计](audit.json)

首窗按时间顺序选择，不按误差挑选。预测计分单位为起点×预测步，重叠时段可能出现多次，不能据此求园区累计排放。
缺失标签不填补，合法0保留；后续模型必须对齐起点、目标时间、信息集与共同评分掩码，并报告覆盖率。
本结果不自动写入项目实验总表；检查通过后人工追加，避免把验证/技术冒烟误写成最终测试结论。
'''
        with (output / 'summary.md').open('x', encoding='utf-8') as stream:
            stream.write(summary)
        manifest = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
        write_json(output / 'artifact_manifest.json', manifest)
    except Exception as exc:
        write_json(output / 'failed.json', {'error': str(exc), 'status': 'failed_partial_output_preserved'})
        raise
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=DEFAULT_HISTORY)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--lookback', type=int, choices=(168,), default=168)
    parser.add_argument('--horizon', type=int, choices=(6, 24), default=24)
    args = parser.parse_args()
    try:
        metrics = run(args.history, args.output_dir, args.split, args.lookback, args.horizon,
                      expected_sha256=INPUT_SHA256)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    status = 'completed' if metrics['overall']['n'] else 'completed_without_scores'
    print(json.dumps({'status': status, 'split': args.split,
                      'output_dir': str(args.output_dir.resolve()), **metrics['overall']}, indent=2))


if __name__ == '__main__':
    main()
