from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import data_formatters.base as base
import libs.utils as utils


BIN_MIN        = 10     # Time resolution (minutes): The duration of each time step
MAX_STEPS      = 72    # Encoder step count: Length of the historical time window used for observation
HORIZON_STEPS  = 12    # Predicted steps: The number of future time steps to be predicted
TOTAL_STEPS    = MAX_STEPS + HORIZON_STEPS
WINDOW_HOURS   = 48    # Data window (hours): Maximum time span retained from admission onwards
SPLIT_SEED     = 42    # Random seed: Used for train/valid/test partitioning
RES_ITEM_IDS   = [615, 618, 220210]  # Excluded item IDs
DYN_DELTA      = 5.0   # Dynamic threshold: Respiratory rate variation threshold for determining whether a window is 'dynamic'
EPS            = 1e-8  # Numerical Stability Constant
BALANCE_RATIO  = 0.5   # Sampling balance ratio: Stable window = Dynamic window × BALANCE_RATIO
MISSING_THRESHOLD = 0.8  # Missing rate threshold: Exceeding this proportion is deemed excessive missing data.
MAX_GAP_HOURS    = 6
GenericDataFormatter = base.GenericDataFormatter
DataTypes            = base.DataTypes
InputTypes           = base.InputTypes


# -------------------- SafeLabelEncoder --------------------
class SafeLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = pd.Series(y, dtype=str)
        if "-1" not in y.values:
            y = pd.concat([y, pd.Series(["-1"])], ignore_index=True)
        return super().fit(y)

    def transform(self, y):
        y = pd.Series(y, dtype=str)
        y = y.mask(~y.isin(self.classes_), "-1")
        return super().transform(y)



class RespiratoryRateFormatter(GenericDataFormatter):

    _column_definition = [
        ("subject_id", DataTypes.REAL_VALUED, InputTypes.ID),
        ("charttime",  DataTypes.DATE,        InputTypes.TIME),
        ("anchor_age",     DataTypes.REAL_VALUED,  InputTypes.STATIC_INPUT),
        ("gender",         DataTypes.CATEGORICAL,  InputTypes.STATIC_INPUT),
        ("admission_type", DataTypes.CATEGORICAL,  InputTypes.STATIC_INPUT),
        ("race",           DataTypes.CATEGORICAL,  InputTypes.STATIC_INPUT),
        ("weight",         DataTypes.REAL_VALUED,  InputTypes.STATIC_INPUT),
        ("height",         DataTypes.REAL_VALUED,  InputTypes.STATIC_INPUT),
        ("hours_from_admit", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ("itemid",   DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ("valuenum", DataTypes.REAL_VALUED,  InputTypes.OBSERVED_INPUT),
        ("respiratory_rate", DataTypes.REAL_VALUED,   InputTypes.TARGET),
    ]

    def __init__(self):
        super().__init__()
        self._time_steps = self.get_fixed_params()["total_time_steps"]
        self._real_scaler:   StandardScaler | None = None
        self._target_scaler: StandardScaler | None = None
        self._real_scalers  : dict[str, StandardScaler] | None = None
        self._target_scalers: dict[str, StandardScaler] | None = None

        self._cat_scalers: dict[str, SafeLabelEncoder] | None = None
        self._num_classes_per_cat_input: list[int] | None = None

        self.train_count = self.valid_count = self.test_count = 0
    def split_data(self, df: pd.DataFrame):
        df = df.copy()
        df["charttime"] = pd.to_datetime(df["charttime"])
        df.sort_values(["subject_id", "charttime"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        first_time = df.groupby("subject_id")["charttime"].transform("min")
        df["hours_from_admit"] = (df["charttime"] - first_time).dt.total_seconds() / 3600.0
        df["step_idx_5m"] = np.floor((df["hours_from_admit"] * 60.0) / BIN_MIN).astype(int)
        df = (
            df.sort_values(["subject_id", "charttime"])
              .drop_duplicates(subset=["subject_id", "step_idx_5m"], keep="last")
        )
        df["charttime"] = first_time + pd.to_timedelta(df["step_idx_5m"] * BIN_MIN, unit="m")
        df["hours_from_admit"] = (df["step_idx_5m"] * BIN_MIN) / 60.0
        df = df[df["hours_from_admit"] <= WINDOW_HOURS]
        if "itemid" in df.columns:
            df = df[~df["itemid"].isin(RES_ITEM_IDS)]
        # df["respiratory_rate"] = pd.to_numeric(df["respiratory_rate"], errors="coerce")
        # df = df.dropna(subset=["respiratory_rate"])

        # for c, dt, _ in self._column_definition:
        #     if c in df.columns:
        #         if dt == DataTypes.REAL_VALUED and c not in ["subject_id", "respiratory_rate"]:
        #             df[c] = df[c].fillna(-1)
        #         elif dt == DataTypes.CATEGORICAL:
        #             df[c] = df[c].fillna("unk")

        if "itemid" in df.columns:
            df = df[~df["itemid"].isin(RES_ITEM_IDS)]

        df["respiratory_rate"] = pd.to_numeric(df["respiratory_rate"], errors="coerce")

        static_obs_cols = [
            c for (c, dt, itype) in self._column_definition
            if itype in (InputTypes.STATIC_INPUT, InputTypes.OBSERVED_INPUT)
        ]
        static_obs_cols = [c for c in static_obs_cols if c in df.columns]

        if static_obs_cols:
            def _patient_missing_rate(g: pd.DataFrame) -> float:
                sub = g[static_obs_cols]
                total = sub.shape[0] * sub.shape[1]
                missing = sub.isna().sum().sum()
                return missing / max(total, 1)

            miss_rate = df.groupby("subject_id", group_keys=False).apply(_patient_missing_rate)
            bad_ids = miss_rate[miss_rate > MISSING_THRESHOLD].index
            if len(bad_ids):
                df = df[~df["subject_id"].isin(bad_ids)]

        df = df.sort_values(["subject_id", "charttime"])

        real_time_cols = [
            c for (c, dt, itype) in self._column_definition
            if dt == DataTypes.REAL_VALUED
            and itype in (InputTypes.KNOWN_INPUT, InputTypes.OBSERVED_INPUT, InputTypes.TARGET)
            and c in df.columns
            and c not in ["subject_id"]
        ]
        MAX_GAP_STEPS    = int(MAX_GAP_HOURS * 60 / BIN_MIN)
        if real_time_cols:
            def _ffill_limit(g: pd.DataFrame) -> pd.DataFrame:
                return g.sort_values("charttime").assign(
                    **{
                        col: g.sort_values("charttime")[col].ffill(limit=MAX_GAP_STEPS)
                        for col in real_time_cols
                    }
                )

            df = df.groupby("subject_id", group_keys=False).apply(_ffill_limit)

        df = df.dropna(subset=["respiratory_rate"])

        for c, dt, _ in self._column_definition:
            if c not in df.columns:
                continue

            if dt == DataTypes.REAL_VALUED and c not in ["subject_id", "respiratory_rate"]:
                median_val = df[c].median()
                df[c] = df[c].fillna(median_val)
            elif dt == DataTypes.CATEGORICAL:
                df[c] = df[c].fillna("unk")

        T = self.get_fixed_params()["total_time_steps"]
        lens = df.groupby("subject_id")["charttime"].size()
        ok_subjects = set(lens[lens >= T].index.astype(df["subject_id"].dtype))
        df = df[df["subject_id"].isin(ok_subjects)]
        if df.empty or len(ok_subjects) == 0:
            need_hours = (T * BIN_MIN) / 60.0
            raise ValueError(
                f"No subjects have length >= total_time_steps ({T} steps ≈ {need_hours:.1f} hours). "
                f"Consider increasing WINDOW_HOURS or reducing MAX_STEPS/HORIZON_STEPS."
            )
        rng = np.random.default_rng(SPLIT_SEED)
        ids = np.array(sorted(ok_subjects))
        rng.shuffle(ids)
        n = len(ids)
        train_ids = ids[: int(0.7 * n)]
        valid_ids = ids[int(0.7 * n): int(0.9 * n)]
        test_ids  = ids[int(0.9 * n):]

        train_raw = df[df.subject_id.isin(train_ids)].reset_index(drop=True)
        valid     = df[df.subject_id.isin(valid_ids)].reset_index(drop=True)
        test      = df[df.subject_id.isin(test_ids )].reset_index(drop=True)

        train = self._build_balanced_train_windows(train_raw)

        self.set_scalers(train)
        train, valid, test = (self.transform_inputs(d) for d in (train, valid, test))
        def _count_windows(d: pd.DataFrame, T_steps: int) -> int:
            if d.empty:
                return 0
            sizes = d.groupby("subject_id")["charttime"].size()
            return int(np.maximum(sizes - T_steps + 1, 0).sum())

        self.train_count = _count_windows(train, T)
        self.valid_count = _count_windows(valid, T)
        self.test_count  = _count_windows(test,  T)
        print("--- Balanced Train Summary ---")
        print(f"[train rows]={len(train)}, [valid rows]={len(valid)}, [test rows]={len(test)}")
        print(f"[train subjects]={train['subject_id'].nunique()}, "
              f"[valid subjects]={valid['subject_id'].nunique()}, "
              f"[test subjects]={test['subject_id'].nunique()}")
        print(f"[train windows]={self.train_count}, [valid windows]={self.valid_count}, [test windows]={self.test_count}")

        for d in (train, valid, test):
            for aux in ("step_idx_5m", "orig_subject_id", "_win_start"):
                if aux in d.columns:
                    d.drop(columns=[aux], inplace=True)

        return train, valid, test

    def _build_balanced_train_windows(self, df_train: pd.DataFrame) -> pd.DataFrame:
        if df_train.empty:
            return df_train

        enc, hor, T = MAX_STEPS, HORIZON_STEPS, TOTAL_STEPS
        pieces = []
        dyn_total = stab_total = 0

        rng = np.random.default_rng(SPLIT_SEED)

        for sid, g in df_train.groupby("subject_id", sort=False):
            g = g.sort_values("charttime").reset_index(drop=True)
            n = len(g)
            if n < T:
                continue

            alb = g["respiratory_rate"].to_numpy()
            dyn_starts, stab_starts = [], []

            for i in range(0, n - T + 1):
                hs = i + enc
                he = i + T
                if he > n:
                    break
                future = alb[hs:he]
                if np.isnan(future).any():
                    continue
                if (np.max(future) - np.min(future)) > DYN_DELTA:
                    dyn_starts.append(i)
                else:
                    stab_starts.append(i)

            if not dyn_starts and not stab_starts:
                continue

            k_dyn  = len(dyn_starts)
            k_stab = int(np.round(k_dyn * BALANCE_RATIO))
            if len(stab_starts) > k_stab:
                stab_pick = rng.choice(stab_starts, size=k_stab, replace=False).tolist()
            else:
                stab_pick = stab_starts

            win_starts = dyn_starts + stab_pick
            dyn_total  += len(dyn_starts)
            stab_total += len(stab_pick)

            for i in win_starts:
                seg = g.iloc[i : i + T].copy()

                new_sid = f"{sid}__win{i}"
                seg["orig_subject_id"] = seg["subject_id"]
                seg["subject_id"]      = new_sid
                seg["_win_start"]      = i 
                pieces.append(seg)

        if not pieces:
            print("[WARN] Balanced sampling produced empty train set; fallback to original train.")
            return df_train

        out = pd.concat(pieces, axis=0).reset_index(drop=True)
        print(f"[Balance] dynamic windows={dyn_total}, stable windows={stab_total}, "
              f"ratio={(dyn_total)/(max(stab_total,1)):.2f}")
        return out

    def set_scalers(self, train_df: pd.DataFrame):
        col_def = self.get_column_definition()
        real_cols = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, col_def, {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        )
        target_col = utils.get_single_col_by_input_type(InputTypes.TARGET, col_def)

        self._real_scaler = StandardScaler().fit(train_df[real_cols]) if len(real_cols) else None
        if self._real_scaler is not None and hasattr(self._real_scaler, "scale_"):
            zero_var_mask = (self._real_scaler.scale_ == 0.0)
            if np.any(zero_var_mask):
                self._real_scaler.scale_[zero_var_mask] = 1.0

        self._target_scaler = StandardScaler().fit(train_df[[target_col]].dropna())
        if hasattr(self._target_scaler, "scale_"):
            self._target_scaler.scale_[self._target_scaler.scale_ == 0.0] = 1.0

        self._real_scalers   = {"__GLOBAL__": self._real_scaler} if self._real_scaler else {}
        self._target_scalers = {"__GLOBAL__": self._target_scaler}

        cat_cols = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, col_def, {InputTypes.ID, InputTypes.TIME}
        )
        self._cat_scalers = {c: SafeLabelEncoder().fit(train_df[c].astype(str)) for c in cat_cols}
        self._num_classes_per_cat_input = [len(enc.classes_) for enc in self._cat_scalers.values()]

    # ---- 5) transform_inputs ----
    def transform_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col_def = self.get_column_definition()

        real_cols = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, col_def, {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        )
        cat_cols = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, col_def, {InputTypes.ID, InputTypes.TIME}
        )
        target_col = utils.get_single_col_by_input_type(InputTypes.TARGET, col_def)

        if self._real_scaler and len(real_cols):
            out[real_cols] = self._real_scaler.transform(out[real_cols])
            arr = out[real_cols].to_numpy()
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            out[real_cols] = arr

        out[[target_col]] = self._target_scaler.transform(out[[target_col]].astype(float))

        for c in cat_cols:
            out[c] = self._cat_scalers[c].transform(out[c].astype(str))

        return out

    def format_predictions(self, preds: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in preds.columns if c not in {"forecast_time", "identifier"}]
        preds = preds.copy()
        preds[cols] = self._target_scaler.inverse_transform(preds[cols])
        return preds

    def get_fixed_params(self):
        return {
            "total_time_steps": TOTAL_STEPS, 
            "num_encoder_steps": MAX_STEPS,
            "num_epochs": 300,
            "early_stopping_patience": 10,
            "multiprocessing_workers": 8,
        }

    def get_default_model_params(self):
        return {
            "hidden_layer_size": 128,
            "dropout_rate": 0.3,
            "learning_rate": 0.00001,
            "minibatch_size": 64,
            "max_gradient_norm": 1.0,
            "num_heads": 6,
            "stack_size": 2,
        }

    def get_num_samples_for_calibration(self):
        return self.train_count, self.valid_count

    def save_scalers(self, folder: str):
        scaler_folder = os.path.join(folder, "scalers")
        os.makedirs(scaler_folder, exist_ok=True)

        scalers = {
            'real_scalers': {
                identifier: {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
                for identifier, scaler in (self._real_scalers or {}).items()
                if scaler is not None
            },
            'cat_scalers': {
                column: enc.classes_.tolist()
                for column, enc in (self._cat_scalers or {}).items()
            },
            'target_scalers': {
                identifier: {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
                for identifier, scaler in (self._target_scalers or {}).items()
            }
        }

        filepath = os.path.join(scaler_folder, 'all_scalers.json')
        with open(filepath, 'w') as f:
            json.dump(scalers, f, indent=4)
        print(f"Scalers have been saved successfully to {filepath}.")

    def load_and_set_scalers(self, folder: str):
        scaler_folder = os.path.join(folder, "scalers")
        filepath = os.path.join(scaler_folder, 'all_scalers.json')
        with open(filepath, 'r') as f:
            scalers = json.load(f)
        self._real_scalers = {}
        real_info = scalers.get('real_scalers', {})
        if "__GLOBAL__" in real_info:
            info = real_info["__GLOBAL__"]
            sc = StandardScaler()
            sc.mean_ = np.array(info['mean'])
            sc.scale_ = np.array(info['scale'])
            self._real_scalers["__GLOBAL__"] = sc
            self._real_scaler = sc
        else:
            self._real_scaler = None
        self._target_scalers = {}
        for identifier, info in scalers.get('target_scalers', {}).items():
            scaler = StandardScaler()
            scaler.mean_ = np.array(info['mean'])
            scaler.scale_ = np.array(info['scale'])
            self._target_scalers[identifier] = scaler
        self._target_scaler = self._target_scalers.get('__GLOBAL__')
        self._cat_scalers = {}
        for column, classes in scalers.get('cat_scalers', {}).items():
            encoder = SafeLabelEncoder()
            encoder.classes_ = np.array(classes)
            self._cat_scalers[column] = encoder
        self._num_classes_per_cat_input = [len(enc.classes_) for enc in self._cat_scalers.values()]

        print(f"Scalers loaded and set successfully from {filepath}.")


