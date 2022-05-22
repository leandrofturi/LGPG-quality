# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import numpy as np
import pandas as pd

################################
# load data
################################

df = pd.read_parquet("datasets/EuropeanSoccerDatabase.parquet")
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
c = "birthday"
values = pd.to_datetime(
    df.loc[valid_rows[c], c], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)
resp = ~values.isna()
results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

c = "date"
values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce")
resp = ~values.isna()
results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False


# RAN_ACC ######################
c = "height"
resp = df.loc[valid_rows[c], c] <= 230
results["ACC"]["RAN_ACC"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

c = "weight"
resp = df.loc[valid_rows[c], c] <= 250
results["ACC"]["RAN_ACC"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False


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
values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce")
resp = (values >= min_date) & (values <= max_date)
results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

# ACC_SEMAN ####################

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


################################
# uniqueness (unicidade) UNI
################################

# UNI_REG ######################
u = "player_name"
columns_uni = ["birthday"]
for c in columns_uni:
    mask = valid_rows[c] & valid_rows[u]
    r = []
    for i in df[u].unique():
        values = df.loc[(df[u] == i) & mask, c]
        if len(values) > 1:
            resp = ~values.ne(values.shift().bfill())
            r.append(resp.sum() / len(values.index))
    results["UNI"]["UNI_REG"][c] = np.mean(r)

################################
# finaly
################################

final = {
    "COMP": np.mean(list(results["COMP"]["COMP_REG"].values())),
    "ACC": (
        np.mean(list(results["ACC"]["ACC_SINT"].values()))
        * np.mean(list(results["ACC"]["RAN_ACC"].values()))
        * np.mean(list(results["ACC"]["ACC_SEMAN"].values()))
    ),
    "CRED": np.mean(list(results["CRED"]["CRED_VAL_DAT"].values())),
    "CONS": np.mean(list(results["CONS"]["CONS_SEMAN"].values())),
    "CURR": np.mean(list(results["CURR"]["CURR_UPD"].values())),
    "UNI": np.mean(list(results["UNI"]["UNI_REG"].values())),
}
