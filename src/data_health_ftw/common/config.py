from dataclasses import dataclass

GHO_BASE_URL = "https://ghoapi.azureedge.net/api/"
REQUEST_TIMEOUT_SECONDS = 30
PAUSE_BETWEEN_REQUESTS_SECONDS = 2


@dataclass(frozen=True)
class Indicator:
    """Um indicador da GHO e os valores de dimensão que selecionam sua série por país.

    Uma dimensão definida como None deve ser nula na linha de origem.
    """

    code: str
    name: str
    dim1: str | None = None
    dim2: str | None = None
    dim3: str | None = None


INDICATORS: tuple[Indicator, ...] = (
    Indicator("NCDMORT3070", "ncd_mortality_30_70", "SEX_BTSX", "AGEGROUP_YEARS30-69"),
    Indicator("SDGPM25", "pm25_exposure", "RESIDENCEAREATYPE_TOTL"),
    Indicator("WHOSIS_000001", "life_expectancy", "SEX_BTSX"),
    Indicator("WHOSIS_000002", "healthy_life_expectancy", "SEX_BTSX"),
    Indicator("MDG_0000000029", "mdg_indicator"),
    Indicator("HIV_0000000026", "hiv_indicator"),
    Indicator("NCD_GLUC_04", "high_glucose_prev", "SEX_BTSX", "AGEGROUP_YEARS18-PLUS"),
    Indicator("BP_04", "hypertension_prev", "SEX_BTSX", "AGEGROUP_YEARS30-79"),
    Indicator("NCD_BMI_30C", "obesity_prev", "SEX_BTSX", "AGEGROUP_YEARS18-PLUS"),
    Indicator("M_Est_smk_curr_std", "smoking_prev", "SEX_BTSX"),
    Indicator("WSH_WATER_SAFELY_MANAGED", "safe_water_pct", "RESIDENCEAREATYPE_TOTL"),
    Indicator("WSH_SANITATION_SAFELY_MANAGED", "safe_sanitation_pct", "RESIDENCEAREATYPE_TOTL"),
    Indicator("NUTRITION_ANT_HAZ_NE2", "child_stunting_prev"),
    Indicator(
        "NUTRITION_ANAEMIA_REPRODUCTIVEAGE_PREV",
        "anaemia_prev_women",
        "SEX_FMLE",
        "PREGNANCYSTATUS_TOTAL",
    ),
    Indicator("GHED_GGHE-D_pc_US_SHA2011", "gov_health_spend_per_capita_usd"),
)

INDICATOR_CODES: list[str] = [i.code for i in INDICATORS]
INDICATOR_NAMES: list[str] = [i.name for i in INDICATORS]
EXPECTED_INDICATOR_COUNT: int = len(INDICATORS)

BRONZE_SCHEMA = "workspace.bronze"
BRONZE_TABLE = f"{BRONZE_SCHEMA}.gho_raw"

SILVER_SCHEMA = "workspace.silver"
SILVER_TABLE = f"{SILVER_SCHEMA}.gho_clean"
SILVER_REJECTED_TABLE = f"{SILVER_SCHEMA}.gho_rejected"

GOLD_SCHEMA = "workspace.gold"
GOLD_LATEST_TABLE = f"{GOLD_SCHEMA}.gho_latest"
GOLD_WIDE_TABLE = f"{GOLD_SCHEMA}.gho_wide"

SILVER_REQUIRED_COLUMNS = ["country_code", "region", "year", "indicator_code", "numeric_value"]
SILVER_UNIQUE_KEY = ["country_code", "year", "indicator_code"]

GOLD_LATEST_REQUIRED_COLUMNS = ["country_code", "indicator", "year", "numeric_value"]
GOLD_LATEST_UNIQUE_KEY = ["country_code", "indicator"]
GOLD_WIDE_ID_COLUMNS = ["country_code", "region", "year"]
GOLD_WIDE_REQUIRED_COLUMNS = ["country_code", "year"]
GOLD_WIDE_UNIQUE_KEY = ["country_code", "year"]


# --- WHO Global Health Estimates (GHE): mortes padronizadas por idade, carregada por arquivo

GHE_VOLUME_DIR = "/Volumes/workspace/bronze/raw_files"
GHE_ASDR_FILE = "age-standardized-death-rates-2000-2023_bycountry_2026-10-01.xlsx"
GHE_ASDR_PATH = f"{GHE_VOLUME_DIR}/{GHE_ASDR_FILE}"
GHE_SEX = "Persons"  # também consta "Males" e "Females"


@dataclass(frozen=True)
class GheCause:
    """Uma causa de morte do GHE (linhas da planilha) e o nome da coluna usado adiante."""

    code: int
    name: str


GHE_CAUSES: tuple[GheCause, ...] = (
    GheCause(1100, "cvd_asdr"),  # mortes cardiovasculares
    GheCause(1130, "ihd_asdr"),  # doença cardíaca isquêmica
    GheCause(1140, "stroke_asdr"),  # avc
    GheCause(1170, "resp_total_asdr"),  # doenças respiratórias totais = doença pulmonar obstrutiva crônica + asma + outras
    GheCause(1180, "copd_asdr"),  # doença pulmonar obstrutiva crônica
    GheCause(1190, "asthma_asdr"),  # asma
    GheCause(1200, "other_resp_asdr"),  # outras doenças respiratórias
    GheCause(390, "lri_asdr"),  # infecções respiratórias inferiores
)
GHE_CAUSE_NAMES: list[str] = [c.name for c in GHE_CAUSES]
GHE_EXPECTED_YEARS = 8  # o arquivo só contém os anos 2000, 2010, 2015 e 2019-2023

# Alvo: mortalidade respiratória crônica = doença pulmonar obstrutiva crônica + asma
# (taxas padronizadas por idade são aditivas).
CHRONIC_RESP_TARGET = "chronic_resp_asdr"
CHRONIC_RESP_COMPONENTS = ("copd_asdr", "asthma_asdr")

# Variáveis em gold.gho_wide que já contêm as causas do target (ou derivam da mortalidade
# por todas as causas). Não serão usadas no modelo, para evitar vazamento do alvo.
LEAKAGE_RISK_FEATURES = ("ncd_mortality_30_70", "life_expectancy", "healthy_life_expectancy")

GHE_BRONZE_TABLE = f"{BRONZE_SCHEMA}.ghe_asdr_raw"
GHE_SILVER_TABLE = f"{SILVER_SCHEMA}.ghe_asdr_clean"
GOLD_ML_FEATURES_TABLE = f"{GOLD_SCHEMA}.ml_features"

GHE_SILVER_REQUIRED_COLUMNS = ["country_code", "year", "ghe_code", "cause_name", "asdr_per_100k"]
GHE_SILVER_UNIQUE_KEY = ["country_code", "year", "ghe_code"]

GOLD_ML_ID_COLUMNS = ["country_code", "region", "year"]
GOLD_ML_TARGET_COLUMNS = GHE_CAUSE_NAMES + [CHRONIC_RESP_TARGET]
GOLD_ML_REQUIRED_COLUMNS = ["country_code", "year"] + GOLD_ML_TARGET_COLUMNS
GOLD_ML_UNIQUE_KEY = ["country_code", "year"]


ML_TARGETS = (CHRONIC_RESP_TARGET, "cvd_asdr")
ML_MIN_YEAR = 2010  # pm25_exposure não tem dados em 2000
ML_FEATURES = (
    "pm25_exposure",
    "obesity_prev",
    "smoking_prev",
    "safe_water_pct",
    "safe_sanitation_pct",
    "anaemia_prev_women",
    "gov_health_spend_per_capita_usd",
)

ML_FILL_FROM_PAST = ("smoking_prev",)

ML_EXCLUDED_FEATURES = {
    "ncd_mortality_30_70": "contains the target causes (cardiovascular, respiratory): leakage",
    "life_expectancy": "derived from all-cause mortality: leakage",
    "healthy_life_expectancy": "derived from all-cause mortality: leakage",
    "high_glucose_prev": "only has data for 2000 and 2010 (100% null from 2015)",
    "hypertension_prev": "only has data up to 2019 (100% null for 2020-2023)",
    "child_stunting_prev": "70-87% null in every year",
    "mdg_indicator": "vague definition and ~24% null in every year",
    "hiv_indicator": "vague definition and ~25% null in every year",
}