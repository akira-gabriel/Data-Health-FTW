"""Parse the WHO GHE age-standardized death rates workbook into a long DataFrame.

Workbook layout (one sheet per year, named "ASDR <year>"):
  * row 7: country names from column H onwards, row 8: ISO-3 codes, row 9: country code flag
  * rows 10+: one row per (sex block, cause); column A = sex, column B = GHE code,
    the cause label sits in the last filled cell of columns C..F
"""
import re
from pathlib import Path

import openpyxl
import pandas as pd

SHEET_PATTERN = re.compile(r"^ASDR (\d{4})$")
FIRST_DATA_COLUMN = 7  # zero-based index of column H
HEADER_NAMES_ROW = 7
HEADER_ISO_ROW = 8
HEADER_FLAG_ROW = 9
FIRST_DATA_ROW = 10
MAX_COLUMNS = 400

OUTPUT_COLUMNS = [
    "country_code",
    "country_name",
    "country_flag",
    "year",
    "sex",
    "ghe_code",
    "cause",
    "asdr_per_100k",
]


def _cause_label(row: tuple) -> str | None:
    labels = [c for c in row[2:6] if c is not None]
    return str(labels[-1]).strip() if labels else None


def _parse_sheet(ws, year: int) -> list[dict]:
    # The sheets declare 16384 columns; real data stops near column 190, so cap the read.
    rows = list(ws.iter_rows(min_row=1, max_col=MAX_COLUMNS, values_only=True))
    names, iso, flags = rows[HEADER_NAMES_ROW - 1], rows[HEADER_ISO_ROW - 1], rows[HEADER_FLAG_ROW - 1]
    last_col = max(i for i, v in enumerate(iso) if v is not None) + 1
    countries = [
        (i, str(iso[i]).strip(), str(names[i]).strip(), flags[i])
        for i in range(FIRST_DATA_COLUMN, last_col)
    ]

    records: list[dict] = []
    for row in rows[FIRST_DATA_ROW - 1 :]:
        sex, ghe_code = row[0], row[1]
        if sex is None or ghe_code is None:  # skips the population row and blank/footnote rows
            continue
        cause = _cause_label(row)
        for col_idx, code, name, flag in countries:
            value = row[col_idx] if col_idx < len(row) else None
            records.append(
                {
                    "country_code": code,
                    "country_name": name,
                    "country_flag": None if flag is None else int(flag),
                    "year": year,
                    "sex": str(sex).strip(),
                    "ghe_code": int(ghe_code),
                    "cause": cause,
                    "asdr_per_100k": None if value is None else float(value),
                }
            )
    return records


def parse_asdr_workbook(path: str | Path) -> pd.DataFrame:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    records: list[dict] = []
    for ws in workbook.worksheets:
        match = SHEET_PATTERN.match(ws.title)
        if match:
            records.extend(_parse_sheet(ws, int(match.group(1))))
    if not records:
        raise ValueError("no 'ASDR <year>' sheets found in workbook")
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
