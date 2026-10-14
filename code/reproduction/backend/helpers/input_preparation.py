from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd
from tqdm.auto import tqdm

def table_path(chunks_root: str | Path, table_name: str) -> Path:
    return Path(chunks_root) / table_name

def list_data_files(path: str | Path, max_files: int | None=None) -> list[Path]:
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob('*.parquet')) + sorted(path.glob('*.csv'))
    if max_files is not None:
        files = files[:int(max_files)]
    if not files:
        raise FileNotFoundError(f'No parquet/csv files found at {path}')
    return files

def read_data_file(path: str | Path, columns: list[str] | None=None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path, columns=columns)
    if path.suffix == '.csv':
        if columns is None:
            return pd.read_csv(path)
        return pd.read_csv(path, usecols=lambda col: col in set(columns))
    raise ValueError(f'Unsupported data file: {path}')

def iter_table(path: str | Path, columns: list[str] | None=None, max_files: int | None=None, desc: str | None=None) -> Iterable[tuple[Path, pd.DataFrame]]:
    files = list_data_files(path, max_files=max_files)
    for file_path in tqdm(files, desc=desc or Path(path).name):
        yield (file_path, read_data_file(file_path, columns=columns))
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:
    Chem = None
