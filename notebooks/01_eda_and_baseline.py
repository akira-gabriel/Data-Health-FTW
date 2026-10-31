# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # EDA e baseline
# MAGIC
# MAGIC ## Perguntas a serem respondidas
# MAGIC 1. Como estão os targets: mortalidade respiratória crônica e cardiovascular, padronizadas por idade?
# MAGIC 2. Que relações os indicadores têm entre eles?
# MAGIC 3. Um baseline model consegue fazer boas previsões?
# MAGIC
# MAGIC Decisões de modelagem:
# MAGIC
# MAGIC - Fora por falta de dados: `high_glucose_prev` (só 2000 e 2010), `hypertension_prev` (só até 2019), `child_stunting_prev` (70–87% nulo), `mdg_indicator` e `hiv_indicator`.
# MAGIC - Será usados os anos 2010 a 2023 por conta da ausência do PM2.5.
# MAGIC - A exclusão por leakage target de ncd_mortality_30_70, life_expectancy e healthy_life_expectancy, o preenchimento do tabagismo em 2019 e 2021 e a validação por país e por região.

# COMMAND ----------

import sys

PROJECT_ROOT = "/Workspace/Users/gakira.t1@gmail.com/Data-Health-FTW"
sys.path.insert(0, f"{PROJECT_ROOT}/src")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_health_ftw.common import config
from data_health_ftw.ml.baseline import (
    cross_validate_grouped,
    grouped_permutation_importance,
    make_models,
)
from data_health_ftw.ml.dataset import build_modeling_frame

# COMMAND ----------

raw = spark.table(config.GOLD_ML_FEATURES_TABLE).toPandas()
print(raw.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cobertura dos indicadores

# COMMAND ----------

coverage = (
    raw.groupby("year")[list(config.ML_FEATURES)]
    .agg(lambda s: s.isna().mean())
    .round(2)
    .T
)
display(coverage.reset_index().rename(columns={"index": "feature"}))

# COMMAND ----------

excluded = pd.DataFrame(
    {"feature": list(config.ML_EXCLUDED_FEATURES), "reason": list(config.ML_EXCLUDED_FEATURES.values())}
)
display(excluded)

# COMMAND ----------

frame = build_modeling_frame(
    raw,
    config.ML_FEATURES,
    config.ML_TARGETS,
    config.ML_MIN_YEAR,
    config.ML_FILL_FROM_PAST,
)
FEATURES = list(config.ML_FEATURES) + ["year"]
print(f"{len(frame)} rows, {frame['country_code'].nunique()} countries, years {sorted(frame['year'].unique())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Target's

# COMMAND ----------

display(frame[list(config.ML_TARGETS)].describe().round(1).reset_index())

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for col, target in enumerate(config.ML_TARGETS):
    axes[0, col].hist(np.log1p(frame[target]), bins=30)
    axes[0, col].set_title(f"{target} (log1p)")
    frame.groupby("year")[target].median().plot(ax=axes[1, col], marker="o")
    axes[1, col].set_title(f"{target}: median by year")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Relação simples entre indicadores e alvos (correlação de Spearman)

# COMMAND ----------

corr = (
    frame[FEATURES + list(config.ML_TARGETS)]
    .corr(method="spearman")
    .loc[FEATURES, list(config.ML_TARGETS)]
    .round(2)
)
display(corr.reset_index().rename(columns={"index": "feature"}))

# COMMAND ----------

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(frame["pm25_exposure"], frame[config.CHRONIC_RESP_TARGET], s=8, alpha=0.4)
ax.set_yscale("log")
ax.set_xlabel("pm25_exposure")
ax.set_ylabel(config.CHRONIC_RESP_TARGET)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Modelo de base com validação por grupo
# MAGIC
# MAGIC `mean_baseline` prevê sempre a média do treino: é o piso. Um modelo só tem valor se ficar **abaixo** dele em MAE/RMSE e **acima** em R².

# COMMAND ----------

results = pd.concat(
    [
        cross_validate_grouped(frame, FEATURES, target, group, make_models())
        for target in config.ML_TARGETS
        for group in ("country_code", "region")
    ],
    ignore_index=True,
)
display(results.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quais indicadores mais ajudam? (importância por permutação em países não vistos)

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, target in zip(axes, config.ML_TARGETS):
    importance = grouped_permutation_importance(
        frame, FEATURES, target, "country_code", make_models()["hist_gbm"]
    )
    importance.sort_values().plot.barh(ax=ax, title=f"{target}: increase in MAE when shuffled")
plt.tight_layout()
plt.show()
