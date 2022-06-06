# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import json
import numpy as np
import pandas as pd

from anonymization.supression import Supression
from anonymization.randomization import Randomization
from anonymization.generalization import Generalization
from anonymization.pseudoanonymization import PseudoAnonymization


################################
# cleaner
################################


def cleaner(df, out_filename):
    print(f"Starting {out_filename}...")

    valid_rows = pd.DataFrame(True, index=df.index, columns=df.columns)
    results = {
        "COMP": {"COMP_REG": {}},
        "ACC": {"ACC_SINT": {}, "RAN_ACC": {}, "ACC_SEMAN": {}},
        "CRED": {"CRED_VAL_DAT": {}},
        "CONS": {"CONS_SEMAN": {}},
        "CURR": {"CURR_UPD": {}},
        "UNI": {"UNI_REG": {}},
    }

    ################################
    # completeness (completude) COMP
    ################################

    # COMP_REG
    for c in df.columns:
        resp = ~df.loc[valid_rows[c], c].isna()
        results["COMP"]["COMP_REG"][c] = resp.sum() / valid_rows[c].sum()
        valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # accuracy (acurácia) ACC
    ################################

    def to_numeric(col):
        return pd.to_numeric(col, errors="coerce")

    # ACC_SINT #####################
    c = "birthday"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "date"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce"
    )
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    # RAN_ACC ######################

    min_date = pd.to_datetime("01012008", format="%d%m%Y")
    max_date = pd.to_datetime("31122016", format="%d%m%Y")

    c = "birthday"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    resp = values <= max_date
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "date"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce"
    )
    resp = (values >= min_date) & (values <= max_date)
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    # ACC_SEMAN ####################
    c = "height"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 230
    results["ACC"]["ACC_SEMAN"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "weight"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 250
    results["ACC"]["ACC_SEMAN"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "birthday"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    resp = (max_date - values).dt.days <= 365 * 120
    results["ACC"]["ACC_SEMAN"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # credibility (credibilidade) CRED
    ################################

    # CRED_VAL_DAT #################

    ################################
    # consistency (consistência) CONS
    ################################

    # CONS_SEMAN ###################

    ################################
    # currentness (atualidade) CURR
    ################################

    # CURR_UPD #####################

    def check_all_years(col):
        values = pd.to_datetime(col, format="%Y-%m-%d", errors="coerce")
        r = pd.Series(range(values.dt.year.min(), values.dt.year.max()), dtype=int)
        resp = r.isin(values.dt.year)
        return sum(resp) == len(resp.index)

    c = "date"
    resp = (
        df.loc[valid_rows[c], ["player_name", c]]
        .groupby("player_name")[c]
        .apply(check_all_years)
    )
    results["CURR"]["CURR_UPD"][c] = resp.sum() / len(resp.index)
    for p in df["player_name"].unique():
        valid_rows.loc[df.loc[(df.player_name == p) & bool(resp.get(p))].index, c] = False

    ################################
    # uniqueness (unicidade) UNI
    ################################

    # UNI_REG ######################
    u = "player_name"
    columns_uni = ["birthday"]
    for c in columns_uni:
        resp = df.groupby(u)[c].apply(
            lambda x: len(x.loc[valid_rows[c] & valid_rows[u]].unique()) <= 1
        )
        results["UNI"]["UNI_REG"][c] = resp.sum() / (len(df[u].unique()))
        # for p in df[u].unique():
        #     valid_rows.loc[df.loc[(df[u] == p) & bool(resp.get(p))].index, c] = False

    ################################
    # finaly
    ################################

    final = {
        "COMP": np.nanmean(list(results["COMP"]["COMP_REG"].values())),
        "ACC": np.nanprod(
            [
                np.nanmean(list(results["ACC"]["ACC_SINT"].values())),
                np.nanmean(list(results["ACC"]["RAN_ACC"].values())),
                np.nanmean(list(results["ACC"]["ACC_SEMAN"].values())),
            ]
        ),
        "CRED": np.nanmean(list(results["CRED"]["CRED_VAL_DAT"].values())),
        "CONS": np.nanmean(list(results["CONS"]["CONS_SEMAN"].values())),
        "CURR": np.nanmean(list(results["CURR"]["CURR_UPD"].values())),
        "UNI": np.nanmean(list(results["UNI"]["UNI_REG"].values())),
    }

    with open(out_filename, "w") as f:
        json.dump(final, f, indent=4, sort_keys=False)


################################
# LGPD columns
################################


LGPD_COLUMNS = [
    "birthday",
    "height",
    "weight",
    "overall_rating",
    "potential",
    "preferred_foot",
    "attacking_work_rate",
    "defensive_work_rate",
    "crossing",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "volleys",
    "dribbling",
    "curve",
    "free_kick_accuracy",
    "long_passing",
    "ball_control",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "balance",
    "shot_power",
    "jumping",
    "stamina",
    "strength",
    "long_shots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "marking",
    "standing_tackle",
    "sliding_tackle",
    "gk_diving",
    "gk_handling",
    "gk_kicking",
    "gk_positioning",
    "gk_reflexes",
]


rules = {
    "height": {"type": "hist", "nbins": 10},
    "weight": {"type": "hist", "nbins": 10},
    "overall_rating": {"type": "hist", "nbins": 8},
    "potential": {"type": "hist", "nbins": 8},
    "crossing": {"type": "hist", "nbins": 8},
    "finishing": {"type": "hist", "nbins": 8},
    "heading_accuracy": {"type": "hist", "nbins": 8},
    "short_passing": {"type": "hist", "nbins": 8},
    "volleys": {"type": "hist", "nbins": 8},
    "dribbling": {"type": "hist", "nbins": 8},
    "curve": {"type": "hist", "nbins": 8},
    "free_kick_accuracy": {"type": "hist", "nbins": 8},
    "long_passing": {"type": "hist", "nbins": 8},
    "ball_control": {"type": "hist", "nbins": 8},
    "acceleration": {"type": "hist", "nbins": 8},
    "sprint_speed": {"type": "hist", "nbins": 8},
    "agility": {"type": "hist", "nbins": 8},
    "reactions": {"type": "hist", "nbins": 8},
    "balance": {"type": "hist", "nbins": 8},
    "shot_power": {"type": "hist", "nbins": 8},
    "jumping": {"type": "hist", "nbins": 8},
    "stamina": {"type": "hist", "nbins": 8},
    "strength": {"type": "hist", "nbins": 8},
    "long_shots": {"type": "hist", "nbins": 8},
    "aggression": {"type": "hist", "nbins": 8},
    "interceptions": {"type": "hist", "nbins": 8},
    "positioning": {"type": "hist", "nbins": 8},
    "vision": {"type": "hist", "nbins": 8},
    "penalties": {"type": "hist", "nbins": 8},
    "marking": {"type": "hist", "nbins": 8},
    "standing_tackle": {"type": "hist", "nbins": 8},
    "sliding_tackle": {"type": "hist", "nbins": 8},
    "gk_diving": {"type": "hist", "nbins": 8},
    "gk_handling": {"type": "hist", "nbins": 8},
    "gk_kicking": {"type": "hist", "nbins": 8},
    "gk_positioning": {"type": "hist", "nbins": 8},
    "gk_reflexes": {"type": "hist", "nbins": 8},
}

################################
# run
################################

df = pd.read_parquet("datasets/EuropeanSoccerDatabase.parquet")
cleaner(df, "output/EuropeanSoccerDatabase_raw.json")
cleaner(
    Supression.anonymize(df, LGPD_COLUMNS),
    "output/EuropeanSoccerDatabase_supression.json",
)
cleaner(
    Randomization.anonymize(df, LGPD_COLUMNS),
    "output/EuropeanSoccerDatabase_randomization.json",
)
cleaner(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    "output/EuropeanSoccerDatabase_generalization.json",
)
cleaner(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    "output/EuropeanSoccerDatabase_pseudoanonymization.json",
)
