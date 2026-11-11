# Data Health FTW

**English** | [Português (BR)](README.pt-BR.md)

Pipeline built on Databricks using a medallion architecture over WHO health data (GHO indicators and Global Health Estimates), with machine learning models to predict cardiovascular and respiratory mortality.

![Databricks](https://img.shields.io/badge/Databricks-FF3621?logo=databricks&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-E25A1C?logo=apachespark&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)

## Objective

Assess how strongly risk factors such as air pollution (PM2.5), smoking, obesity, sanitation and public health spending relate to age-standardized mortality rates from cardiovascular and respiratory causes across countries, and build models that predict those rates.

This is a data engineering project with machine learning, following a pipeline of ingestion, cleaning, feature engineering, validation and modeling, all native to Databricks.

## Architecture

```mermaid
flowchart LR
    A["GHO API<br/>risk factor indicators"] --> B[("Bronze<br/>raw tables")]
    C["GHE xlsx<br/>age-standardized mortality rates"] --> B
    B --> S[("Silver<br/>cleaned tables")]
    S --> G[("Gold<br/>gho_wide, ml_features")]
    G --> V["Feature validation"]
    V --> M["Machine learning"]
```

Job `ml_features_pipeline`:

`ghe-bronze-ingest` → `ghe-silver-transform` → `ml-features-build` → `ml-features-validate`

| Layer | Table | Content |
|---|---|---|
| Bronze | `workspace.bronze.ghe_asdr_raw` | Raw GHE mortality rates, loaded from the xlsx file |
| Silver | `workspace.silver.ghe_asdr_clean` | Table organized by country x year x cause |
| Gold | `workspace.gold.gho_wide` | GHO indicators, one row per country and year |
| Gold | `workspace.gold.ml_features` | GHE targets left-joined with `gho_wide` (1,480 rows) |

## Data sources

| Source | Use | Notes |
|---|---|---|
| [GHO OData API (WHO)](https://www.who.int/data/gho/info/gho-odata-api) | Risk factor indicators (PM2.5, smoking, obesity, ...) | Discontinued by the WHO. It returned HTTP 500 on Oct 30, 2026. |
| WHO Global Health Estimates 2023 (xlsx) | Age-standardized mortality rates, per 100,000, by cause, sex and country | Available years: 2000, 2010, 2015, 2019, 2020, 2021, 2022, 2023 |
| [Xmart API (WHO)](https://xmart-api-public.who.int/DATA_/$metadata) | Replacement for the GHO API (`RELAY_*` entities + `REFMART.REF_COUNTRY` for M49 → ISO-3 and region) | **In use** in the bronze ingest |

Panel size: up to 185 countries x 8 years = 1,480 rows. The modeling window is 2010-2023 (1,295 rows).

## Engineering decisions

- **API discontinuation.** The GHO OData API was discontinued, so the cause-of-death targets come from the GHE file and the indicators are now ingested from the WHO Xmart API. Xmart flags some series as `DRAFT` (obesity, child malnutrition and the health spending series). The obesity values were checked against the old data and are identical.
- **No target leakage.** `ncd_mortality_30_70` is left out because it includes the target itself, as are `life_expectancy` and `healthy_life_expectancy`. Pollution-attributable mortality is not used as a target because it is computed from the same exposures used as features.
- **Sparse features dropped.** `child_stunting_prev` (70-87% null in every year), `high_glucose_prev` (only 2000 and 2010), `hypertension_prev` (only up to 2019) and `mdg_indicator` / `hiv_indicator` (~25% null and vague definitions).
- **Derived spending feature.** `gov_health_spend_per_capita_usd` is no longer published by Xmart, so it is rebuilt as current health expenditure per capita (US$) times the share of domestic government spending in current expenditure (both in national currency). Checked against the old values: mean absolute difference of US$0.02 on a mean of US$675.
- **Revised smoking estimates.** `smoking_prev` differs only slightly from the old values, with a mean absolute difference of about 1 percentage point concentrated in a few countries and no systematic bias. The series includes WHO projections for 2025 and 2030.
- **The modeling window starts in 2010.** `pm25_exposure` is 100% null in 2000.
- **Gaps in `smoking_prev`.** 2019 and 2021 are entirely missing, so the last observed value is carried forward within each country (past values only, to avoid leakage that would hurt the machine learning models).

## Machine learning

- **Targets:** `cvd_asdr` (cardiovascular, GHE 1100; `ihd_asdr` as a narrower alternative) and `chronic_resp_asdr` (COPD + asthma). Sensitivity checks use COPD only and total respiratory diseases (1170).
- **Validation:** `GroupKFold` by country (and by region group). A random row split would put the same country in both train and test, since countries change little from one year to the next, which inflates the metrics.
- **Models:** simple baseline vs. final model, compared in a metrics table, plus variable importance.
- **Experiment tracking:** MLflow (planned).
- **Limitations:**
  - Only 8 years of data per country.
  - Data is aggregated by country, so conclusions do not apply to individuals.
  - Correlation is not causation, so no causal argument should be assumed.

## Roadmap

- ✅ Migrate the indicator ingestion from the GHO API to the Xmart API
- ⬜ Compare grouped CV metrics with and without `year`
- ⬜ Baseline vs. final model, with variable importance
- ⬜ Track experiments with MLflow
- ⬜ Add results and insights to this README