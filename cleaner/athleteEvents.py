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


def div(x, y):
    if y == 0:
        return np.nan
    return x / y


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
        results["COMP"]["COMP_REG"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # accuracy (acurácia) ACC
    ################################

    def to_numeric(col):
        return pd.to_numeric(col, errors="coerce")

    # ACC_SINT #####################
    c = "year"
    values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y", errors="coerce")
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    cities = pd.read_csv("utils/cities.csv")

    c = "city"
    resp = df.loc[valid_rows[c], c].isin(cities.city)
    results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    DICT = {
        "sex": ["M", "F"],
        "medal": ["Gold", "Silver", "Bronze"],
        "season": ["Summer", "Winter"],
    }
    for c in DICT.keys():
        resp = df.loc[valid_rows[c], c].isin(DICT[c])
        results["ACC"]["ACC_SINT"][c] = div(resp.sum(), valid_rows[c].sum())
        valid_rows.loc[resp.loc[~resp].index, c] = False

    # RAN_ACC ######################

    c = "year"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 2016
    results["ACC"]["RAN_ACC"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    # ACC_SEMAN ####################

    c = "age"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 120
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "height"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 230
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "weight"
    resp = to_numeric(df.loc[valid_rows[c], c]) <= 250
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    olympics = pd.read_csv("utils/olympicsCities.csv")

    c = "city"
    resp = df.loc[valid_rows[c], c].isin(olympics.city)
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "year"
    resp = df.loc[valid_rows[c], c].isin(olympics.year)
    results["ACC"]["ACC_SEMAN"][c] = div(resp.sum(), valid_rows[c].sum())
    valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # credibility (credibilidade) CRED
    ################################

    # CRED_VAL_DAT #################

    ################################
    # consistency (consistência) CONS
    ################################

    # CONS_SEMAN ###################
    mask = valid_rows.year & valid_rows.games
    r1 = df.loc[mask, "year"] == df.loc[mask, "games"].str[:4].astype(int)
    results["CONS"]["CONS_SEMAN"]["year#games"] = div(r1.sum(), mask.sum())
    valid_rows.loc[r1.loc[~r1].index, "year"] = False
    valid_rows.loc[r1.loc[~r1].index, "games"] = False

    ################################
    # currentness (atualidade) CURR
    ################################

    # CURR_UPD #####################

    ################################
    # uniqueness (unicidade) UNI
    ################################

    # UNI_REG ######################
    u = "id"
    columns_uni = ["name", "sex"]
    for c in columns_uni:
        resp = df.groupby(u)[c].apply(
            lambda x: len(x.index)
            if len(x.loc[valid_rows[c] & valid_rows[u]].unique()) <= 1
            else -len(x.index)
        )
        results["UNI"]["UNI_REG"][c] = div(resp[resp > 0].sum(), abs(resp).sum())
        for p in df[u].unique():
            valid_rows.loc[df.loc[(df[u] == p) & ((resp.get(p) or 1) < 0)].index, c] = False

    ################################
    # finaly
    ################################

    final = {
        "COMP": np.mean(list(results["COMP"]["COMP_REG"].values())),
        "ACC": np.prod(
            [
                np.mean(list(results["ACC"]["ACC_SINT"].values())),
                np.mean(list(results["ACC"]["RAN_ACC"].values())),
                np.mean(list(results["ACC"]["ACC_SEMAN"].values())),
            ]
        ),
        "CRED": -1,
        "CONS": np.mean(list(results["CONS"]["CONS_SEMAN"].values())),
        "CURR": -1,
        "UNI": np.mean(list(results["UNI"]["UNI_REG"].values())),
    }

    with open(out_filename, "w") as f:
        json.dump(final, f, indent=4, sort_keys=False)


################################
# LGPD columns
################################

LGPD_COLUMNS = [
    "name",
    "sex",
    "age",
    "height",
    "weight",
    "team",
    "games",
    "year",
    "city",
    "sport",
    "event",
    "medal",
    "region",
]

rules = {
    "name": {"type": "split", "char": " ", "keep": 0},
    "event": {"type": "split", "char": " ", "keep": 0},
    "age": {"type": "hist", "nbins": 10},
    "height": {"type": "hist", "nbins": 10},
    "weight": {"type": "hist", "nbins": 10},
}


################################
# run
################################

df = pd.read_parquet("datasets/athleteEvents.parquet")
cleaner(df, "output/athleteEvents_raw.json")
cleaner(Supression.anonymize(df, LGPD_COLUMNS), "output/athleteEvents_supression.json")
cleaner(
    Randomization.anonymize(df, LGPD_COLUMNS), "output/athleteEvents_randomization.json"
)
cleaner(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    "output/athleteEvents_generalization.json",
)
cleaner(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    "output/athleteEvents_pseudoanonymization.json",
)
