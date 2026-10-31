from collections.abc import Sequence

import pandas as pd


def fill_from_past(
    df: pd.DataFrame,
    columns: Sequence[str],
    group_column: str = "country_code",
    order_column: str = "year",
) -> pd.DataFrame:
    """Propaga o último valor observado em cada grupo, em ordem de tempo (só valores do passado)."""
    columns = list(columns)
    if not columns:
        return df.copy()
    ordered = df.sort_values([group_column, order_column]).copy()
    ordered[columns] = ordered.groupby(group_column)[columns].ffill()
    return ordered


def build_modeling_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    targets: Sequence[str],
    min_year: int,
    fill_from_past_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Seleciona linhas e colunas de modelagem a partir da tabela ml_features da gold

    Preenche dados antes do filtro de ano, para que anos anteriores à janela possam preencher lacunas dos anos seguintes
    """
    required = {"country_code", "region", "year", *features, *targets}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"columns missing from ml_features: {sorted(missing)}")

    filled = fill_from_past(df, fill_from_past_columns)
    frame = filled[filled["year"] >= min_year].dropna(subset=list(targets)).copy()
    frame["region"] = frame["region"].fillna("UNKNOWN")
    columns = ["country_code", "region", "year", *features, *targets]
    return frame[columns].reset_index(drop=True)
