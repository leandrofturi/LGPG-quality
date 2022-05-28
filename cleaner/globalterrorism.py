# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import json
import numpy as np
import pandas as pd
import dask.dataframe as dd
from geosky import geo_plug
from autocorrect import Speller
from geopy.geocoders import Nominatim

################################
# load data
################################

df = pd.read_parquet("datasets/globalterrorism.parquet")
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
# "bussiness rules"
df.loc[df["imonth"] == 0, "imonth"] = None
df.loc[df["iday"] == 0, "iday"] = None

for c in df.columns:
    resp = ~df.loc[valid_rows[c], c].isna()
    results["COMP"]["COMP_REG"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False


################################
# accuracy (acurácia) ACC
################################

# ACC_SINT #####################
VALUES_DATES = {"iyear": "%Y", "imonth": "%m", "iday": "%d", "resolution": "%d/%m/%Y"}
for c in VALUES_DATES.keys():
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format=VALUES_DATES[c], errors="coerce"
    )
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

VALUES_COUNTRIES = [
    "country_txt",
    "natlty1_txt",
    "kidhijcountry",
    "natlty2_txt",
    "natlty3_txt",
]
countries = geo_plug.all_CountryNames()
for c in VALUES_COUNTRIES:
    resp = df.loc[valid_rows[c], c].isin(countries)
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

# "region_txt"

# states = geo_plug.all_Country_StateNames()
# states = json.loads(states)
# states = sum([sum(list(s.values()), []) for s in states], [])

# c = "provstate"
# resp = df.loc[valid_rows[c], c].isin(states)
# results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
# valid_rows.loc[resp.loc[~resp].index, c] = False

# cities = geo_plug.all_State_CityNames('all')
# cities = json.loads(cities)
# cities = sum([sum(list(c.values()), []) for c in cities], [])

# c = "city"
# resp = df.loc[valid_rows[c], c].isin(cities)
# results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
# valid_rows.loc[resp.loc[~resp].index, c] = False


# RAN_ACC ######################

min_date = pd.to_datetime("01011970", format="%d%m%Y")
max_date = pd.to_datetime("31122017", format="%d%m%Y")
for c in VALUES_DATES.keys():
    values = pd.to_datetime(
        df.loc[valid_rows[c], c], format=VALUES_DATES[c], errors="coerce"
    )
    resp = (values >= min_date) & (values <= max_date)
    results["ACC"]["RAN_ACC"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

# ACC_SEMAN ####################

spell_columns = [
    "summary",
    "alternative_txt",
    "attacktype1_txt",
    "attacktype2_txt",
    "attacktype3_txt",
    "targtype1_txt",
    "targsubtype1_txt",
    "targtype2_txt",
    "targsubtype2_txt",
    "targtype3_txt",
    "targsubtype3_txt",
    "claimmode_txt",
    "claimmode2_txt",
    "claimmode3_txt",
    "weaptype1_txt",
    "weapsubtype1_txt",
    "weaptype2_txt",
    "weapsubtype2_txt",
    "weaptype3_txt",
    "weapsubtype3_txt",
    "weaptype4_txt",
    "weapsubtype4_txt",
    "propextent_txt",
    "hostkidoutcome_txt",
    "corp1" "corp2" "corp3" "motive" "divert" "addnotes",
    "scite1",
    "scite2",
    "scite3",
]

spell = Speller(lang='en', fast=True)
for c in spell_columns:
    resp = [spell(row) == row for row in df.loc[valid_rows[c], c]]
    results["ACC"]["ACC_SEMAN"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

geolocator = Nominatim(user_agent="tcc")


def func(row):
    try:
        location = geolocator.reverse(f'{row["latitude"]}, {row["longitude"]}')
        return (
            row["country_txt"] == location.raw["address"].get("country"),
            row["city"] == location.raw["address"].get("city"),
        )
    except:
        return (None, None)


df_dd = dd.from_pandas(df, npartitions=1)
resp = df_dd.apply(func, axis=1, meta=(None, 'object')).compute()


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
    results["UNI"]["UNI_REG"][c] = np.nanmean(r)

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
