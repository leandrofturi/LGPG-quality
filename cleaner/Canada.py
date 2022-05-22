# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import re
import numpy as np
import pandas as pd

################################
# load data
################################

df = pd.read_parquet("datasets/Canada.parquet")
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
c = "date_of_birth"
values = pd.to_datetime(
    df.loc[valid_rows[c], c].str[:10], format="%d/%m/%Y", errors="coerce"
).fillna(
    pd.to_datetime(
        df.loc[valid_rows[c], c].str[:10], format="%Y-%m-%d", errors="coerce"
    )
)
resp = ~values.isna()
results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

c = "date_of_death"
values = pd.to_datetime(
    df.loc[valid_rows[c], c].str[:10], format="%d/%m/%Y", errors="coerce"
).fillna(
    pd.to_datetime(
        df.loc[valid_rows[c], c].str[:10], format="%Y-%m-%d", errors="coerce"
    )
)
resp = ~values.isna()
results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

# RAN_ACC ######################
max_date = pd.to_datetime("31122022", format="%d%m%Y")
c = "date_of_birth"
values = pd.to_datetime(
    df.loc[valid_rows[c], c].str[:10], format="%d/%m/%Y", errors="coerce"
).fillna(
    pd.to_datetime(
        df.loc[valid_rows[c], c].str[:10], format="%Y-%m-%d", errors="coerce"
    )
)
resp = ~values.isna()
results["ACC"]["RAN_ACC"][c] = (values <= max_date).sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

c = "date_of_death"
values = pd.to_datetime(
    df.loc[valid_rows[c], c].str[:10], format="%d/%m/%Y", errors="coerce"
).fillna(
    pd.to_datetime(
        df.loc[valid_rows[c], c].str[:10], format="%Y-%m-%d", errors="coerce"
    )
)
resp = ~values.isna()
results["ACC"]["RAN_ACC"][c] = (values <= max_date).sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

# ACC_SEMAN ####################
c = "years_of_service"
values = pd.Series(
    [
        pd.to_numeric(v and v[0], errors="coerce")
        for v in df.loc[valid_rows[c], c].str.split(" days")
    ]
)
results["ACC"]["ACC_SEMAN"][c] = (values <= 40 * 365).sum() / valid_rows[
    c
].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False


################################
# credibility (credibilidade) CRED
################################

# CRED_VAL_DAT #################


################################
# consistency (consistência) CONS
################################

# CONS_SEMAN ###################
mask = valid_rows.date_of_birth & valid_rows.date_of_death & valid_rows.years_of_service
c = "date_of_birth"
dt_1 = pd.to_datetime(
    df.loc[mask, c].str[:10], format="%d/%m/%Y", errors="coerce"
).fillna(
    pd.to_datetime(
        df.loc[mask, c].str[:10], format="%Y-%m-%d", errors="coerce"
    )
)
c = "date_of_death"
dt_2 = pd.to_datetime(
    df.loc[mask, c].str[:10], format="%d/%m/%Y", errors="coerce"
).fillna(
    pd.to_datetime(
        df.loc[mask, c].str[:10], format="%Y-%m-%d", errors="coerce"
    )
)
c = 'years_of_service'
dt_3 = pd.Series(
    [
        pd.to_numeric(v and v[0], errors="coerce")
        for v in df.loc[mask, c].str.split(" days")
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
columns_uni = ['date_of_birth', 'place_of_birth']
for c in columns_uni:
    mask = valid_rows[c] & valid_rows[u]
    r = []
    for i in df[u].unique():
        values = df.loc[(df[u] == i) & mask, c]
        if len(values) > 1:
            resp = values.ne(values.shift().bfill())
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
