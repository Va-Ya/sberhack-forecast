"""Fit a frozen selection-only recipe and export native inference artifacts."""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast.contracts import CUTOFF, next_monday
from forecast.data import Panel, load_archive, sha256_file
from forecast.features import CATEGORICAL, baselines, make_encoder
from forecast.metrics import compute_metrics
from forecast.model import (
    TrainingConfig, calibrated_predictions, forecast_frame, model_predictions, training_matrix,
)


def fit_recipe(panel, cutoff, recipe, directory, threads=2, reuse=None):
    directory.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(**{**recipe['training'], 'threads': threads})
    encoder = make_encoder(panel, cutoff)
    models, backends, transforms = {}, {}, {}
    matrix = target = None
    columns = None
    start = time.perf_counter()
    reuse_config = json.loads((reuse/'config.json').read_text()) if reuse else {}
    for name, member in recipe['members'].items():
        backends[name] = member.get('backend', 'lightgbm')
        transforms[name] = member.get('transform', 'identity')
        if reuse and name in {'log', 'raw'} and (reuse/f'{name}.txt').is_file():
            # Reuse only the exact incumbent family, cutoff and training sample.
            expected = {**asdict(config), 'iterations': member['rounds']}
            if (reuse_config['cutoff'] != str(cutoff.date()) or
                    any(reuse_config['training'].get(k) != v for k, v in expected.items() if k != 'threads') or
                    reuse_config['encoder'] != encoder):
                raise ValueError('Incumbent identity does not match requested training')
            columns = reuse_config['features']
            models[name] = lgb.Booster(model_file=str(reuse/f'{name}.txt'))
            print(f'Reused exact incumbent {name}', flush=True)
        else:
            if matrix is None:
                matrix, target = training_matrix(panel, cutoff, encoder, config)
                columns = list(matrix.columns)
                print(f'Training matrix: {matrix.shape}', flush=True)
            y = np.log1p(target) if transforms[name] == 'log1p' else target
            if backends[name] == 'catboost':
                from catboost import CatBoostRegressor

                data = matrix.copy()
                data[CATEGORICAL] = data[CATEGORICAL].astype(np.int32)
                models[name] = CatBoostRegressor(**member['params'], iterations=member['rounds'],
                                                 thread_count=threads)
                models[name].fit(data, y, cat_features=CATEGORICAL, verbose=False)
            else:
                dataset = lgb.Dataset(matrix, label=y, categorical_feature=CATEGORICAL)
                params = {**member['params'], 'num_threads': threads}
                models[name] = lgb.train(params, dataset, num_boost_round=member['rounds'])
            print(f'Fitted {name}', flush=True)
        suffix = 'cbm' if backends[name] == 'catboost' else 'txt'
        models[name].save_model(str(directory/f'{name}.{suffix}'))
    metadata = {'version': '2.0.0', 'selected': 'calibrated', 'models': list(models),
                'model_backends': backends, 'model_transforms': transforms,
                'calibration': recipe['calibration'], 'encoder': encoder, 'features': columns,
                'threads': threads, 'training': asdict(config), 'cutoff': str(cutoff.date()),
                'training_recipe': recipe, 'audit': panel.audit,
                'dependencies': {'python': platform.python_version(), 'lightgbm': lgb.__version__}}
    (directory/'config.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    panel.pairs.to_csv(directory/'pairs.csv', index=False)
    learned = model_predictions(models, encoder, columns, panel, cutoff, threads, backends, transforms)
    values = calibrated_predictions(baselines(panel, cutoff), learned, recipe['calibration'])
    frame = forecast_frame(values[:, panel.output_indices], panel.output_pairs, next_monday(cutoff))
    frame.to_csv(directory.parent/'forecast.csv', index=False, float_format='%.8f')
    report = {'cutoff': str(cutoff.date()), 'elapsed_seconds': time.perf_counter()-start,
              'rows': len(frame), 'recipe': recipe,
              'forecast_sha256': sha256_file(directory.parent/'forecast.csv')}
    if cutoff + pd.Timedelta(days=63) <= panel.dates[-1]:
        truth = np.stack([panel.total(w, w+pd.Timedelta(days=6))
                          for w in pd.date_range(next_monday(cutoff), periods=9, freq='7D')])
        report['metrics'] = compute_metrics(truth[:, panel.output_indices], values[:, panel.output_indices])
        np.savez_compressed(directory.parent/'predictions.npz', truth=truth, forecast=values)
    (directory.parent/'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--directory', required=True, help='New directory for candidate native model')
    parser.add_argument('--cutoff', default=str(CUTOFF.date()))
    parser.add_argument('--archive', help='Original source ZIP; used if --panel is absent')
    parser.add_argument('--panel', help='Preprocessed Panel cache')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--reuse-incumbent', type=Path)
    args = parser.parse_args()
    if not args.panel and not args.archive:
        parser.error('Provide --archive or --panel')
    panel = Panel.load(Path(args.panel)) if args.panel else load_archive(Path(args.archive), CUTOFF)
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff > CUTOFF or cutoff > panel.dates[-1]:
        parser.error('Training cutoff exceeds allowed history')
    recipe = json.loads(Path(args.recipe).read_text(encoding='utf-8'))
    fit_recipe(panel, cutoff, recipe, Path(args.directory), args.threads, args.reuse_incumbent)


if __name__ == '__main__':
    main()
