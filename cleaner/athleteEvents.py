# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import json
import numpy as np
import pandas as pd
from geosky import geo_plug

################################
# load data
################################

df = pd.read_parquet("datasets/athleteEvents.parquet")
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
c = "year"
values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y", errors="coerce")
resp = ~values.isna()
results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

cities = geo_plug.all_State_CityNames("all")
cities = json.loads(cities)
cities = sum([sum([*c.values()], []) for c in cities], [])

# c = "city"
# resp = df.loc[valid_rows[c], c].isin(cities)
# results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
# valid_rows.loc[resp.loc[~resp].index, c] = False

DICT = {"sex": ["M", "F"], "medal": ["Gold", "Silver", "Bronze"]}
for c in DICT.keys():
    resp = df.loc[valid_rows[c], c].isin(DICT[c])
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

c = "year"
resp = df.loc[valid_rows[c], c] <= 2016
results["ACC"]["RAN_ACC"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

# ACC_SEMAN ####################
olympics = pd.read_csv("utils/olympicsCities.csv")
translated_cities = ['Athina', 'Torino', 'Antwerpen', 'Roma', 'Moskva', 'Sankt Moritz']

c = "city"
resp = df.loc[valid_rows[c], c].isin(olympics.city) | df.loc[valid_rows[c], c].isin(translated_cities)
results["ACC"]["ACC_SEMAN"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False


################################
# credibility (credibilidade) CRED
################################

# CRED_VAL_DAT #################

c = "city"
resp = df.loc[valid_rows[c], c].isin(olympics.city) | df.loc[valid_rows[c], c].isin(translated_cities)
results["CRED"]["CRED_VAL_DAT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

c = "year"
resp = df.loc[valid_rows[c], c].isin(olympics.year)
results["CRED"]["CRED_VAL_DAT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False


################################
# consistency (consistência) CONS
################################

# CONS_SEMAN ###################
mask = valid_rows.year & valid_rows.games
r1 = df.loc[mask, "year"] == df.loc[mask, "games"].str[:4].astype(int)
results["CONS"]["CONS_SEMAN"]["year#games"] = r1.sum() / mask.sum()
valid_rows.loc[r1.loc[~r1].index, "mask"] = False
valid_rows.loc[r1.loc[~r1].index, "mask"] = False

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
    mask = valid_rows[c] & valid_rows[u]
    r = []
    for i in df[u].unique():
        values = df.loc[(df[u] == i) & mask, c]
        if len(values) > 0:
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
