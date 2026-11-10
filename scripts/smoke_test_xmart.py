import sys

sys.path.insert(0, "src")

import pandas as pd

from data_health_ftw.bronze import gho_client, transform
from data_health_ftw.common import config

payloads, failed = gho_client.fetch_all_payloads(
    config.INDICATORS,
    config.GHO_BASE_URL,
    config.REQUEST_TIMEOUT_SECONDS,
    config.PAUSE_BETWEEN_REQUESTS_SECONDS,
    config.GHO_PAGE_SIZE,
)
assert not failed, f"failed indicators: {failed}"

countries = gho_client.fetch_country_reference(
    config.REFMART_BASE_URL, config.REQUEST_TIMEOUT_SECONDS, config.GHO_PAGE_SIZE
)
raw_df = transform.build_raw_frame(payloads, config.GHO_COLUMN_MAP)
df = transform.attach_country_reference(raw_df, countries)

summary = []
for indicator in config.INDICATORS:
    mask = df["IndicatorCode"] == indicator.code
    for column, expected in (("Dim1", indicator.dim1), ("Dim2", indicator.dim2), ("Dim3", indicator.dim3)):
        mask &= df[column].isna() if expected is None else df[column] == expected
    selected = df[mask].assign(
        year=lambda d: pd.to_numeric(d["TimeDim"]),
        value=lambda d: pd.to_numeric(d["NumericValue"]),
    )
    summary.append({
        "indicator": indicator.name,
        "rows": len(selected),
        "countries": selected["SpatialDim"].nunique(),
        "first_year": selected["year"].min(),
        "last_year": selected["year"].max(),
        "null_values": int(selected["value"].isna().sum()),
        "duplicate_keys": int(selected.duplicated(["SpatialDim", "year"]).sum()),
    })

print(pd.DataFrame(summary).to_string(index=False))
print("regions:", sorted(df["ParentLocationCode"].dropna().unique()))