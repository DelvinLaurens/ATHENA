import argparse
import os
import pickle
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.brain.live_log_analyzer import load_log, normalize_log, parse_csv, profit_factor


DEFAULT_SIGNAL_SCORE_BANDS = (
    (68.0, 70.0),
    (72.0, 75.0),
)
NUMERIC_FEATURES = [
    'ai_score',
    'vol_pct',
    'change_4h',
    'is_policy_band',
]
CATEGORICAL_FEATURES = [
    'symbol',
    'signal',
    'risk_level',
]


@dataclass
class MetaFilterConfig:
    log_path: str = os.path.join('data', 'predictions_log.csv')
    model_path: str = os.path.join('models', 'meta_filter.pkl')
    threshold: float = 52.0
    min_training_samples: int = 200
    min_symbol_samples: int = 10
    symbol_profit_factor_threshold: float = 1.0
    symbol_win_rate_threshold: float = 45.0
    fee_pct: float = 0.001
    slippage_pct: float = 0.0
    position_size: float = 100.0
    signal_score_bands: tuple[tuple[float, float], ...] = DEFAULT_SIGNAL_SCORE_BANDS
    blacklist_scope: str = 'policy'
    use_model_cache: bool = True


class AthenaMetaFilter:
    def __init__(self, config=None):
        self.config = config or MetaFilterConfig()
        self.pipeline = None
        self.ready = False
        self.source = 'untrained'
        self.training_rows = 0
        self.positive_rate = 0.0
        self.auto_blacklist = set()
        self.symbol_stats = pd.DataFrame()

    @staticmethod
    def _score_in_bands(series, score_bands):
        mask = pd.Series(False, index=series.index)
        for lower, upper in score_bands:
            mask = mask | ((series >= lower) & (series < upper))
        return mask

    def _round_trip_cost_pct(self):
        return (self.config.fee_pct + self.config.slippage_pct) * 2 * 100

    def _load_normalized_log(self):
        raw_df = load_log(self.config.log_path)
        return normalize_log(
            raw_df,
            position_size=self.config.position_size,
            fee_pct=self.config.fee_pct,
            slippage_pct=self.config.slippage_pct,
        )

    def _prepare_features(self, df):
        feature_df = pd.DataFrame(index=df.index)
        feature_df['symbol'] = df.get('symbol', '').astype(str).str.upper()
        feature_df['signal'] = df.get('signal', '').astype(str).str.upper()
        feature_df['risk_level'] = df.get('risk_level', 'UNKNOWN').fillna('UNKNOWN').astype(str).str.upper()
        feature_df['ai_score'] = pd.to_numeric(df.get('ai_score'), errors='coerce')
        feature_df['vol_pct'] = pd.to_numeric(df.get('vol_pct'), errors='coerce')
        feature_df['change_4h'] = pd.to_numeric(df.get('change_4h'), errors='coerce')
        feature_df['is_policy_band'] = self._score_in_bands(
            feature_df['ai_score'],
            self.config.signal_score_bands,
        ).astype(int)
        return feature_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def _prepare_target(self, df):
        target = (pd.to_numeric(df['result_num'], errors='coerce') > 0).astype(int)
        if 'return_pct' not in df.columns:
            return target

        return_pct = pd.to_numeric(df['return_pct'], errors='coerce')
        has_return = return_pct.notna()
        if has_return.any():
            net_return_pct = return_pct - self._round_trip_cost_pct()
            target.loc[has_return] = (net_return_pct.loc[has_return] > 0).astype(int)
        return target

    def _build_pipeline(self):
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ])
        preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, NUMERIC_FEATURES),
            ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
        ])
        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
            )),
        ])

    def _load_cached_model(self):
        if not self.config.use_model_cache or not os.path.exists(self.config.model_path):
            return False

        if os.path.exists(self.config.log_path):
            model_mtime = os.path.getmtime(self.config.model_path)
            log_mtime = os.path.getmtime(self.config.log_path)
            if model_mtime < log_mtime:
                return False

        try:
            with open(self.config.model_path, 'rb') as model_file:
                payload = pickle.load(model_file)
        except (OSError, pickle.PickleError, EOFError):
            return False

        self.pipeline = payload.get('pipeline')
        self.training_rows = int(payload.get('training_rows') or 0)
        self.positive_rate = float(payload.get('positive_rate') or 0.0)
        self.ready = self.pipeline is not None and self.training_rows >= self.config.min_training_samples
        self.source = 'cache' if self.ready else 'untrained'
        return self.ready

    def _save_model(self):
        if self.pipeline is None:
            return

        model_dir = os.path.dirname(self.config.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        payload = {
            'pipeline': self.pipeline,
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'training_rows': self.training_rows,
            'positive_rate': self.positive_rate,
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
        }
        with open(self.config.model_path, 'wb') as model_file:
            pickle.dump(payload, model_file)

    def _build_symbol_blacklist(self, log_df):
        validated_df = log_df[log_df['result_num'].notna()].copy()
        if validated_df.empty:
            self.symbol_stats = pd.DataFrame()
            self.auto_blacklist = set()
            return

        if self.config.blacklist_scope == 'policy':
            policy_mask = self._score_in_bands(
                pd.to_numeric(validated_df['ai_score'], errors='coerce'),
                self.config.signal_score_bands,
            )
            validated_df = validated_df[policy_mask].copy()

        rows = []
        for symbol, group in validated_df.groupby('symbol'):
            return_group = group[group['return_pct'].notna()]
            samples = int(len(group))
            return_samples = int(len(return_group))
            wins = int((group['result_num'] > 0).sum())
            win_rate = (wins / samples) * 100 if samples else 0.0
            net_pnl = float(return_group['pnl'].sum()) if return_samples else 0.0
            pf = profit_factor(return_group)
            rows.append({
                'symbol': symbol,
                'samples': samples,
                'return_samples': return_samples,
                'wins': wins,
                'losses': samples - wins,
                'win_rate_pct': round(win_rate, 2),
                'profit_factor': pf,
                'net_pnl': round(net_pnl, 4),
            })

        self.symbol_stats = pd.DataFrame(rows)
        if self.symbol_stats.empty:
            self.auto_blacklist = set()
            return

        pf_numeric = pd.to_numeric(
            self.symbol_stats['profit_factor'].replace(float('inf'), 999.0),
            errors='coerce',
        ).fillna(0.0)
        blacklist_mask = (
            (self.symbol_stats['samples'] >= self.config.min_symbol_samples)
            & (self.symbol_stats['return_samples'] >= self.config.min_symbol_samples)
            & (self.symbol_stats['net_pnl'] < 0)
            & (
                (pf_numeric < self.config.symbol_profit_factor_threshold)
                | (self.symbol_stats['win_rate_pct'] < self.config.symbol_win_rate_threshold)
            )
        )
        self.auto_blacklist = set(self.symbol_stats.loc[blacklist_mask, 'symbol'])

    def fit_or_load(self):
        log_df = self._load_normalized_log()
        self._build_symbol_blacklist(log_df)
        if self._load_cached_model():
            return self

        train_df = log_df[
            log_df['result_num'].notna()
            & log_df['ai_score'].notna()
        ].copy()
        self.training_rows = int(len(train_df))
        if self.training_rows < self.config.min_training_samples:
            self.ready = False
            self.source = 'insufficient_data'
            return self

        X = self._prepare_features(train_df)
        y = self._prepare_target(train_df)
        if y.nunique() < 2:
            self.ready = False
            self.source = 'one_class'
            self.positive_rate = round(float(y.mean()) * 100, 2)
            return self

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        self.positive_rate = round(float(y.mean()) * 100, 2)
        self.ready = True
        self.source = 'trained'
        self._save_model()
        return self

    def _items_to_frame(self, items):
        rows = []
        for item in items:
            rows.append({
                'symbol': item.get('symbol'),
                'signal': 'LONG' if float(item.get('ai_score', 0.0)) > 50 else 'SHORT',
                'risk_level': item.get('risk_level'),
                'ai_score': item.get('ai_score'),
                'vol_pct': item.get('vol_pct'),
                'change_4h': item.get('change_4h', item.get('change_24h')),
            })
        return pd.DataFrame(rows)

    def predict_scores(self, items):
        if not items:
            return []

        if not self.ready:
            return [50.0 for _ in items]

        item_df = self._items_to_frame(items)
        X = self._prepare_features(item_df)
        probabilities = self.pipeline.predict_proba(X)
        positive_index = list(self.pipeline.named_steps['model'].classes_).index(1)
        return [round(float(probability[positive_index]) * 100, 2) for probability in probabilities]

    def annotate_items(self, items):
        self.fit_or_load()
        scores = self.predict_scores(items)
        for item, score in zip(items, scores):
            symbol = str(item.get('symbol', '')).upper()
            item['meta_score'] = score
            item['meta_filter_ready'] = self.ready
            item['meta_filter_source'] = self.source
            item['is_auto_blacklisted'] = symbol in self.auto_blacklist
        return items

    def blacklist_report(self):
        if self.symbol_stats.empty:
            return pd.DataFrame()

        report_df = self.symbol_stats[self.symbol_stats['symbol'].isin(self.auto_blacklist)].copy()
        if report_df.empty:
            return report_df

        report_df['profit_factor'] = report_df['profit_factor'].apply(
            lambda value: 'inf' if value == float('inf') else round(float(value), 4)
        )
        return report_df.sort_values(['net_pnl', 'win_rate_pct'], ascending=[True, True])

    def summary(self):
        return {
            'ready': self.ready,
            'source': self.source,
            'threshold': self.config.threshold,
            'training_rows': self.training_rows,
            'positive_rate': self.positive_rate,
            'auto_blacklist_count': len(self.auto_blacklist),
            'auto_blacklist_symbols': sorted(self.auto_blacklist),
            'blacklist_scope': self.config.blacklist_scope,
        }


def parse_score_bands(value):
    if not value:
        return DEFAULT_SIGNAL_SCORE_BANDS

    bands = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        lower, upper = item.split(':', maxsplit=1)
        bands.append((float(lower), float(upper)))
    return tuple(bands)


def print_table(title, df, max_rows=None):
    print()
    print(title)
    print('=' * len(title))
    if df.empty:
        print('No rows.')
        return

    display_df = df.head(max_rows) if max_rows else df
    print(display_df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description='ATHENA live-log meta filter')
    parser.add_argument('--log-path', default=os.path.join('data', 'predictions_log.csv'))
    parser.add_argument('--model-path', default=os.path.join('models', 'meta_filter.pkl'))
    parser.add_argument('--threshold', type=float, default=52.0)
    parser.add_argument('--min-training-samples', type=int, default=200)
    parser.add_argument('--min-symbol-samples', type=int, default=10)
    parser.add_argument('--score-bands', default='68:70,72:75')
    parser.add_argument('--blacklist-scope', choices=['policy', 'all'], default='policy')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--top-blacklist', type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    meta_filter = AthenaMetaFilter(MetaFilterConfig(
        log_path=args.log_path,
        model_path=args.model_path,
        threshold=args.threshold,
        min_training_samples=args.min_training_samples,
        min_symbol_samples=args.min_symbol_samples,
        signal_score_bands=parse_score_bands(args.score_bands),
        blacklist_scope=args.blacklist_scope,
        use_model_cache=not args.no_cache,
    ))
    meta_filter.fit_or_load()
    summary = meta_filter.summary()

    print('ATHENA Meta Filter')
    print('==================')
    print(f"Ready: {summary['ready']}")
    print(f"Source: {summary['source']}")
    print(f"Threshold: {summary['threshold']}")
    print(f"Training rows: {summary['training_rows']}")
    print(f"Positive rate after fees: {summary['positive_rate']}%")
    print(f"Blacklist scope: {summary['blacklist_scope']}")
    print(f"Auto blacklist count: {summary['auto_blacklist_count']}")

    blacklist_df = meta_filter.blacklist_report()
    print_table('Auto Blacklist', blacklist_df, max_rows=args.top_blacklist)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
        blacklist_df.to_csv(os.path.join(args.output_dir, 'auto_blacklist.csv'), index=False)
        print()
        print(f"Saved meta filter reports to: {args.output_dir}")


if __name__ == '__main__':
    main()
