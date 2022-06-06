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

    # ACC_SINT #####################
    c = "date_and_place_of_birth"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c].str[:10], format="%Y-%m-%d", errors="coerce"
    )
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    # RAN_ACC ######################
    max_date = pd.to_datetime("31122022", format="%d%m%Y")
    c = "date_and_place_of_birth"
    values = pd.to_datetime(
        df.loc[valid_rows[c], c].str[:10], format="%Y-%m-%d", errors="coerce"
    )
    resp = ~values.isna()
    results["ACC"]["RAN_ACC"][c] = (values <= max_date).sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

    # ACC_SEMAN ####################

    c = "number_of_votes"
    values = pd.Series([
        pd.to_numeric(re.sub(r",", ".", re.sub(r"\(|\)|%", "", v)), errors="coerce")
        for v in df.loc[valid_rows[c], c].str[-7:]
    ])
    results["ACC"]["ACC_SEMAN"][c] = (values <= 100).sum() / valid_rows[c].sum()
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

    ################################
    # uniqueness (unicidade) UNI
    ################################

    # UNI_REG ######################
    u = "title"
    columns_uni = ["date_and_place_of_birth"]
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
    "date_and_place_of_birth",
    "occupation",
    "education",
    "won_in_the_elections",
    "election_committee",
    "name",
    "club_circle",
    "district",
]

rules = {
    "date_and_place_of_birth": {"type": "split", "char": " ", "keep": -1},
    "occupation": {"type": "split", "char": " ", "keep": 0},
    "name": {"type": "split", "char": " ", "keep": 0},
}

################################
# run
################################

df = pd.read_parquet("datasets/Poland.parquet")
cleaner(df, "output/Poland_raw.json")
cleaner(Supression.anonymize(df, LGPD_COLUMNS), "output/Poland_supression.json")
cleaner(Randomization.anonymize(df, LGPD_COLUMNS), "output/Poland_randomization.json")
cleaner(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    "output/Poland_generalization.json",
)
cleaner(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    "output/Poland_pseudoanonymization.json",
)
