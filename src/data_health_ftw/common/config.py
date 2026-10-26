from dataclasses import dataclass

GHO_BASE_URL = "https://ghoapi.azureedge.net/api/"
REQUEST_TIMEOUT_SECONDS = 30
PAUSE_BETWEEN_REQUESTS_SECONDS = 2


@dataclass(frozen=True)
class Indicator:
    """One GHO indicator and the dimension values that select its country-level series.

    A dimension set to None must be null in the source row.
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