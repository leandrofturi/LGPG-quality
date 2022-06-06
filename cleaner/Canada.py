# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import re
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

    def to_date(col):
        return pd.to_datetime(col.str[:10], format="%d/%m/%Y", errors="coerce").fillna(
            pd.to_datetime(col.str[:10], format="%Y-%m-%d", errors="coerce")
        )

    # ACC_SINT #####################
    c = "date_of_birth"
    values = to_date(df.loc[valid_rows[c], c])
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "date_of_death"
    values = to_date(df.loc[valid_rows[c], c])
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    # RAN_ACC ######################
    max_date = pd.to_datetime("31122022", format="%d%m%Y")
    c = "date_of_birth"
    values = to_date(df.loc[valid_rows[c], c])
    resp = ~values.isna()
    results["ACC"]["RAN_ACC"][c] = (values <= max_date).sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    c = "date_of_death"
    values = to_date(df.loc[valid_rows[c], c])
    resp = ~values.isna()
    results["ACC"]["RAN_ACC"][c] = (values <= max_date).sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    # ACC_SEMAN ####################
    c = "years_of_service"
    values = pd.Series(
        [
            pd.to_numeric(v and v[0], errors="coerce")
            for v in df.loc[valid_rows[c], c].astype(str).str.split(" days")
        ]
    )
    results["ACC"]["ACC_SEMAN"][c] = (values <= 40 * 365).sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    ################################
    # credibility (credibilidade) CRED
    ################################

    # CRED_VAL_DAT #################

    ################################
    # consistency (consistência) CONS
    ################################

    # CONS_SEMAN ###################
    mask = (
        valid_rows.date_of_birth
        & valid_rows.date_of_death
        & valid_rows.years_of_service
    )
    dt_1 = to_date(df.loc[mask, "date_of_birth"])
    dt_2 = to_date(df.loc[mask, "date_of_death"])
    dt_3 = pd.Series(
        [
            pd.to_numeric(v and v[0], errors="coerce")
            for v in df.loc[mask, "years_of_service"].astype(str).str.split(" days")
        ]
    )
    r1 = (dt_2 - dt_1).dt.days.reset_index(drop=True) >= dt_3.reset_index(drop=True)
    results["CONS"]["CONS_SEMAN"]["date_of_birth#date_of_death#years_of_service"] = (
        r1.sum() / mask.sum()
    )
    valid_rows.loc[r1.loc[~r1].index, "date_of_birth"] = False
    valid_rows.loc[r1.loc[~r1].index, "date_of_death"] = False
    valid_rows.loc[r1.loc[~r1].index, "years_of_service"] = False

    ################################
    # currentness (atualidade) CURR
    ################################

    # CURR_UPD #####################

    ################################
    # uniqueness (unicidade) UNI
    ################################

    # UNI_REG ######################
    u = "title"
    columns_uni = ["date_of_birth", "place_of_birth"]
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
    "date_of_birth",
    "place_of_birth",
    "date_of_death",
    "profession",
    "years_of_service",
    "cause_of_death",
]

rules = {
    "place_of_birth": {"type": "split", "char": " ", "keep": -1},
    "profession": {"type": "split", "char": " ", "keep": 0},
    "years_of_service": {"type": "split", "char": " ", "keep": 0},
}

rules_ = {
    "place_of_birth": {"type": "split", "char": " ", "keep": -1},
    "profession": {"type": "split", "char": " ", "keep": 0},
    "years_of_service": {"type": "hist", "nbins": 4},
}

################################
# run
################################

df = pd.read_parquet("datasets/Canada.parquet")
cleaner(df, "output/Canada_raw.json")
cleaner(Supression.anonymize(df, LGPD_COLUMNS), "output/Canada_supression.json")
cleaner(
    Randomization.anonymize(df, LGPD_COLUMNS),
    "output/Canada_randomization.json",
)
df_ = Generalization.anonymize(df, LGPD_COLUMNS, rules)
cleaner(
    Generalization.anonymize(df_, LGPD_COLUMNS, rules_),
    "output/Canada_generalization.json",
)
cleaner(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    "output/Canada_pseudoanonymization.json",
)
