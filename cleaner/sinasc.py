# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import numpy as np
import pandas as pd
from requests import get as GET

################################
# load data
################################

df = pd.read_parquet("datasets/sinasc.parquet")
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
columns_date = [
    "dt_cadastro",
    "dt_declarac",
    "dt_nasc",
    "dt_nasc_mae",
    "dt_recebim",
    "dt_rec_orig",
    "dt_ult_menst",
]
for c in columns_date:
    values = pd.to_datetime(df.loc[valid_rows[c], c], format="%d%m%Y", errors="coerce")
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

r = GET("https://servicodados.ibge.gov.br/api/v1/localidades/municipios")
mun_ibge = [
    {
        "id": m["id"],
        "nome": m["nome"],
        "UF": m["microrregiao"]["mesorregiao"]["UF"]["sigla"],
    }
    for m in r.json()
]
mun_ibge = pd.DataFrame(mun_ibge)

columns_mun_ibge = ["cod_mun_nasc", "cod_mun_natu", "cod_mun_res"]
for c in columns_mun_ibge:
    values = df.loc[valid_rows[c], c].astype(int, errors="ignore").astype(str)
    resp = values.isin(mun_ibge["id"].astype(str).str[:6])
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False


# RAN_ACC ######################
min_date = pd.to_datetime("01011970", format="%d%m%Y")
max_date = pd.to_datetime("31122017", format="%d%m%Y")
for c in ["dt_cadastro", "dt_declarac", "dt_nasc", "dt_recebim", "dt_rec_orig"]:
    values = pd.to_datetime(df.loc[valid_rows[c], c], format="%d%m%Y", errors="coerce")
    results["ACC"]["RAN_ACC"][c] = (
        (values >= min_date) & (values <= max_date)
    ).sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

for c in ["dt_nasc_mae", "dt_ult_menst"]:
    values = pd.to_datetime(df.loc[valid_rows[c], c], format="%d%m%Y", errors="coerce")
    results["ACC"]["RAN_ACC"][c] = (values <= max_date).sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False


# ACC_SEMAN ####################
max_date = pd.to_datetime("31122017", format="%d%m%Y")
columns_date = [
    "dt_nasc",
    "dt_nasc_mae",
    "dt_ult_menst",
]
for c in columns_date:
    values = pd.to_datetime(df.loc[valid_rows[c], c], format="%d%m%Y", errors="coerce")
    results["ACC"]["ACC_SEMAN"][c] = (
        (((max_date - values).dt.days/365) <= 120).sum()
    ) / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False


################################
# credibility (credibilidade) CRED
################################

# CRED_VAL_DAT #################
c = "consultas"
results["CRED"]["CRED_VAL_DAT"][c] = (
    df.loc[valid_rows[c], c] <= 42
).sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

c = "cod_mun_nasc"
results["CRED"]["CRED_VAL_DAT"][c] = (
    df.loc[valid_rows[c], c]
    .isin(mun_ibge.loc[mun_ibge["UF"] == "ES", "id"].astype(str).str[:6])
    .sum()
) / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False


################################
# consistency (consistência) CONS
################################

# CONS_SEMAN ###################
mask = valid_rows.qtd_fil_mort & valid_rows.qtd_fil_vivo & valid_rows.qtd_gest_ant
r1 = (
    df.loc[mask, "qtd_fil_mort"] + df.loc[mask, "qtd_fil_vivo"]
    <= df.loc[mask, "qtd_gest_ant"]
)
results["CONS"]["CONS_SEMAN"]["qtd_fil_mort#qtd_fil_vivo#qtd_gest_ant"] = (
    r1.sum() / mask.sum()
)

mask = valid_rows.qtd_part_ces & valid_rows.qtd_part_nor & valid_rows.qtd_gest_ant
r2 = (
    df.loc[mask, "qtd_part_ces"] + df.loc[mask, "qtd_part_nor"]
    <= df.loc[mask, "qtd_gest_ant"]
)
results["CONS"]["CONS_SEMAN"]["qtd_part_ces#qtd_part_nor#qtd_gest_ant"] = (
    r2.sum() / mask.sum()
)

mask = valid_rows.id_anomal & valid_rows.cod_anomal
r3 = (df.loc[mask, "id_anomal"] == 2.0) & df.loc[mask, "cod_anomal"].isnull()
results["CONS"]["CONS_SEMAN"]["id_anomal#cod_anomal"] = r3.sum() / mask.sum()

mask = valid_rows.dt_nasc & valid_rows.dt_cadastro
dt_1 = pd.to_datetime(df.loc[mask, "dt_nasc"], format="%d%m%Y", errors="coerce")
dt_2 = pd.to_datetime(df.loc[mask, "dt_cadastro"], format="%d%m%Y", errors="coerce")
r4 = dt_1 <= dt_2
results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_cadastro"] = r4.sum() / mask.sum()

mask = valid_rows.dt_nasc & valid_rows.dt_ult_menst
dt_1 = pd.to_datetime(df.loc[mask, "dt_ult_menst"], format="%d%m%Y", errors="coerce")
dt_2 = pd.to_datetime(df.loc[mask, "dt_nasc"], format="%d%m%Y", errors="coerce")
r5 = dt_1 <= dt_2
results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_ult_menst"] = r5.sum() / mask.sum()

mask = valid_rows.dt_nasc & valid_rows.dt_rec_orig
dt_1 = pd.to_datetime(df.loc[mask, "dt_nasc"], format="%d%m%Y", errors="coerce")
dt_2 = pd.to_datetime(df.loc[mask, "dt_rec_orig"], format="%d%m%Y", errors="coerce")
r6 = dt_1 <= dt_2
results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_rec_orig"] = r6.sum() / mask.sum()

mask = valid_rows.dt_nasc & valid_rows.dt_recebim
dt_1 = pd.to_datetime(df.loc[mask, "dt_nasc"], format="%d%m%Y", errors="coerce")
dt_2 = pd.to_datetime(df.loc[mask, "dt_recebim"], format="%d%m%Y", errors="coerce")
r7 = dt_1 <= dt_2
results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_recebim"] = r7.sum() / mask.sum()

mask = valid_rows.dt_nasc & valid_rows.dt_nasc_mae
dt_1 = pd.to_datetime(df.loc[mask, "dt_nasc_mae"], format="%d%m%Y", errors="coerce")
dt_2 = pd.to_datetime(df.loc[mask, "dt_nasc"], format="%d%m%Y", errors="coerce")
r8 = df.loc[mask, "dt_nasc"] <= df.loc[mask, "dt_nasc_mae"]
results["CONS"]["CONS_SEMAN"]["dt_nasc#dt_nasc_mae"] = r8.sum() / mask.sum()

mask = valid_rows.loc_nasc & valid_rows.cod_estab
r9 = (
    (df.loc[mask, "loc_nasc"] == 1.0)
    & (df.loc[mask, "loc_nasc"] == 2.0)
    & (~df.loc[mask, "cod_estab"].isnull())
)
results["CONS"]["CONS_SEMAN"]["loc_nasc#cod_estab"] = r9.sum() / mask.sum()

valid_rows.loc[r1.loc[~r1].index, "qtd_fil_mort"] = False
valid_rows.loc[r1.loc[~r1].index, "qtd_fil_vivo"] = False
valid_rows.loc[(r1 & r2).loc[~(r1 & r2)].index, "qtd_gest_ant"] = False
valid_rows.loc[r2.loc[~r2].index, "qtd_part_ces"] = False
valid_rows.loc[r2.loc[~r2].index, "qtd_part_nor"] = False
valid_rows.loc[r3.loc[~r3].index, "id_anomal"] = False
valid_rows.loc[r3.loc[~r3].index, "cod_anomal"] = False
valid_rows.loc[
    (r4 & r5 & r6 & r7 & r8).loc[~(r4 & r5 & r6 & r7 & r8)].index, "dt_nasc"
] = False
valid_rows.loc[r4.loc[~r4].index, "dt_cadastro"] = False
valid_rows.loc[r5.loc[~r5].index, "dt_ult_menst"] = False
valid_rows.loc[r6.loc[~r6].index, "dt_rec_orig"] = False
valid_rows.loc[r7.loc[~r7].index, "dt_recebim"] = False
valid_rows.loc[r8.loc[~r8].index, "dt_nasc_mae"] = False
valid_rows.loc[r9.loc[~r9].index, "loc_nasc"] = False
valid_rows.loc[r9.loc[~r9].index, "cod_estab"] = False


################################
# currentness (atualidade) CURR
################################

# CURR_UPD #####################
values_dt_recebim = pd.to_datetime(df.dt_recebim, format="%d%m%Y", errors="coerce")
for c in ["dt_cadastro", "dt_declarac", "dt_nasc"]:
    mask = valid_rows[c] & valid_rows.dt_recebim
    values = pd.to_datetime(df.loc[mask, c], format="%d%m%Y", errors="coerce")
    results["CURR"]["CURR_UPD"][c] = (
        sum((values_dt_recebim - values).dt.days <= 365) / mask.sum()
    )
    valid_rows.loc[resp.loc[~resp].index, c] = False


################################
# uniqueness (unicidade) UNI
################################

# UNI_REG ######################


################################
# finaly
################################

final = {
    "COMP": np.nanmean(list(results["COMP"]["COMP_REG"].values())),
    "ACC": np.nanprod([
        np.nanmean(list(results["ACC"]["ACC_SINT"].values())),
        np.nanmean(list(results["ACC"]["RAN_ACC"].values())),
        np.nanmean(list(results["ACC"]["ACC_SEMAN"].values())),
    ]),
    "CRED": np.nanmean(list(results["CRED"]["CRED_VAL_DAT"].values())),
    "CONS": np.nanmean(list(results["CONS"]["CONS_SEMAN"].values())),
    "CURR": np.nanmean(list(results["CURR"]["CURR_UPD"].values())),
    "UNI": np.nanmean(list(results["UNI"]["UNI_REG"].values())),
}
