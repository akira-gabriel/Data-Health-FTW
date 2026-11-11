from dataclasses import dataclass

GHO_BASE_URL = "https://xmart-api-public.who.int/DATA_/"
REFMART_BASE_URL = "https://xmart-api-public.who.int/refmart/"
GHO_DEFAULT_ENTITY = "RELAY_MAY2023"
GHO_PAGE_SIZE = 5000
REQUEST_TIMEOUT_SECONDS = 60
PAUSE_BETWEEN_REQUESTS_SECONDS = 1

GHO_COLUMN_MAP: dict[str, str] = {
    "IND_CODE": "IndicatorCode",
    "DIM_GEO_CODE_TYPE": "SpatialDimType",
    "DIM_GEO_CODE_M49": "GeoM49",  # replaced by ISO-3 (SpatialDim) via REFMART.REF_COUNTRY
    "DIM_TIME_TYPE": "TimeDimType",
    "DIM_TIME": "TimeDim",
    "DIM_1_CODE": "Dim1Type",
    "DIM_MEMBER_1_CODE": "Dim1",
    "DIM_2_CODE": "Dim2Type",
    "DIM_MEMBER_2_CODE": "Dim2",
    "DIM_3_CODE": "Dim3Type",
    "DIM_MEMBER_3_CODE": "Dim3",
    "DIM_PUBLISH_STATE_CODE": "PublishState",
    "VALUE_NUMERIC": "NumericValue",
    "VALUE_NUMERIC_LOWER": "Low",
    "VALUE_NUMERIC_UPPER": "High",
}


@dataclass(frozen=True)
class Indicator:

    code: str
    name: str
    dim1: str | None = None
    dim2: str | None = None
    dim3: str | None = None
    entity: str = GHO_DEFAULT_ENTITY
    publish_states: tuple[str | None, ...] = ("PUBLISHED",)


INDICATORS: tuple[Indicator, ...] = (
    Indicator("NCDMORT3070", "ncd_mortality_30_70", "TOTAL", "YEARS30-69"),
    Indicator("SDGPM25", "pm25_exposure", "TOTAL"),
    Indicator("WHOSIS_000001", "life_expectancy", "TOTAL"),
    Indicator("WHOSIS_000002", "healthy_life_expectancy", "TOTAL"),
    Indicator("MDG_0000000029", "mdg_indicator", entity="RELAY_GHO"),
    Indicator("HIV_0000000026", "hiv_indicator", entity="RELAY_GHO", publish_states=(None,)),
    Indicator("NCD_GLUC_04", "high_glucose_prev", "BTSX", "YEARS18-PLUS", entity="RELAY_GHO"),
    Indicator(
        "NCD_BMI_30C", "obesity_prev", "TOTAL", "Y_GE18",
        entity="RELAY_NUTRITION", publish_states=("DRAFT",),
    ),
    Indicator("M_Est_smk_curr_std", "smoking_prev", "BTSX", entity="RELAY_GHO"),
    Indicator("WSH_WATER_SAFELY_MANAGED", "safe_water_pct", "TOTAL"),
    Indicator("WSH_SANITATION_SAFELY_MANAGED", "safe_sanitation_pct", "TOTAL"),
    Indicator(
        "NUTRITION_ANT_HAZ_NE2", "child_stunting_prev",
        entity="RELAY_NUTRITION", publish_states=("DRAFT",),
    ),
    Indicator("NUTRITION_ANAEMIA_REPRODUCTIVEAGE_PREV", "anaemia_prev_women", "FEMALE", "TOTAL"),
    Indicator("GHED_CHE_PC_USD", "che_per_capita_usd", entity="RELAY_PSCP", publish_states=("DRAFT",)),
    Indicator("GHED_CHE_NCU", "che_ncu", entity="RELAY_PSCP", publish_states=("DRAFT",)),
    Indicator("GHED_GGHED_NCU", "gghed_ncu", entity="RELAY_PSCP", publish_states=("DRAFT",)),
)

INDICATOR_CODES: list[str] = [i.code for i in INDICATORS]
INDICATOR_NAMES: list[str] = [i.name for i in INDICATORS]
GOV_HEALTH_SPEND = "gov_health_spend_per_capita_usd"
DERIVED_FEATURE_NAMES: list[str] = [GOV_HEALTH_SPEND]
WIDE_VALUE_COLUMNS: list[str] = INDICATOR_NAMES + DERIVED_FEATURE_NAMES
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


GHE_VOLUME_DIR = "/Volumes/workspace/bronze/raw_files"
GHE_ASDR_FILE = "age-standardized-death-rates-2000-2023_bycountry_2026-10-01.xlsx"
GHE_ASDR_PATH = f"{GHE_VOLUME_DIR}/{GHE_ASDR_FILE}"
GHE_SEX = "Persons" 


@dataclass(frozen=True)
class GheCause:
    """A GHE cause of death (spreadsheet rows) and the column name used downstream."""

    code: int
    name: str


GHE_CAUSES: tuple[GheCause, ...] = (
    GheCause(1100, "cvd_asdr"),  # cardiovascular deaths
    GheCause(1130, "ihd_asdr"),  # ischaemic heart disease
    GheCause(1140, "stroke_asdr"),  # stroke
    GheCause(1170, "resp_total_asdr"),  # total respiratory diseases = COPD + asthma + other
    GheCause(1180, "copd_asdr"),  # chronic obstructive pulmonary disease (COPD)
    GheCause(1190, "asthma_asdr"),  # asthma
    GheCause(1200, "other_resp_asdr"),  # other respiratory diseases
    GheCause(390, "lri_asdr"),  # lower respiratory infections
)
GHE_CAUSE_NAMES: list[str] = [c.name for c in GHE_CAUSES]
GHE_EXPECTED_YEARS = 8  # the file only has the years 2000, 2010, 2015 and 2019-2023

CHRONIC_RESP_TARGET = "chronic_resp_asdr"
CHRONIC_RESP_COMPONENTS = ("copd_asdr", "asthma_asdr")

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
ML_MIN_YEAR = 2010  # pm25_exposure has no data in 2000
ML_FEATURES = (
    "pm25_exposure",
    "obesity_prev",
    "smoking_prev",
    "safe_water_pct",
    "safe_sanitation_pct",
    "anaemia_prev_women",
    GOV_HEALTH_SPEND,
)

ML_FILL_FROM_PAST = ("smoking_prev",)

ML_EXCLUDED_FEATURES = {
    "ncd_mortality_30_70": "contains the target causes (cardiovascular, respiratory): leakage",
    "life_expectancy": "derived from all-cause mortality: leakage",
    "healthy_life_expectancy": "derived from all-cause mortality: leakage",
    "high_glucose_prev": "only has data for 2000 and 2010 (100% null from 2015)",
    "hypertension_prev": "BP_04 is not published in the WHO Xmart API (had data only up to 2019)",
    "child_stunting_prev": "70-87% null in every year",
    "mdg_indicator": "vague definition and ~24% null in every year",
    "hiv_indicator": "vague definition and ~25% null in every year",
}