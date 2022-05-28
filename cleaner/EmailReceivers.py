# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import numpy as np
import pandas as pd

################################
# load data
################################

df1 = pd.read_parquet("datasets/EmailReceivers1.parquet")
df2 = pd.read_parquet("datasets/EmailReceivers2.parquet")
df = df1.append(df2)
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
    "metadata_date_sent",
    "metadata_date_released",
    "extracted_date_sent",
    "extracted_date_released",
]
for c in columns_date:
    values = pd.to_datetime(df.loc[valid_rows[c], c], errors="coerce")
    resp = ~values.isna()
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False


# RAN_ACC ######################
for c in columns_date:
    values = pd.to_datetime(df.loc[valid_rows[c], c], errors="coerce")
    resp = values.dt.year <= 2015
    results["ACC"]["RAN_ACC"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

# ACC_SEMAN ####################


################################
# credibility (credibilidade) CRED
################################

# CRED_VAL_DAT #################

text_columns = [
    "alias",
    "extracted_subject",
    "extracted_to",
    "extracted_from",
    "extracted_cc",
    "extracted_date_sent",
    "extracted_case_number",
    "extracted_doc_number",
    "extracted_date_released",
    "extracted_body_text",
]
raw_text = df["raw_text"]
for c in text_columns:
    resp = df.loc[valid_rows[c], [c, "raw_text"]].apply(
        lambda x: x[c] in x.raw_text if x[c] else False, axis=1
    )
    results["CRED"]["CRED_VAL_DAT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False


################################
# consistency (consistência) CONS
################################

# CONS_SEMAN ###################
mask = (
    valid_rows.extracted_to
    & valid_rows.metadata_to
    & df.extracted_release_in_part_or_full
)
r1 = (
    (df.loc[mask, "extracted_to"] == df.loc[mask, "metadata_to"])
    | (df.loc[mask, "metadata_to"] == "H")
) & (df.loc[mask, "extracted_release_in_part_or_full"] == "RELEASE IN FULL")
results["CONS"]["CONS_SEMAN"]["extracted_to#metadata_to"] = r1.sum() / mask.sum()

mask = (
    valid_rows.metadata_from
    & valid_rows.extracted_from
    & df.extracted_release_in_part_or_full
)
r2 = (
    (
        df.loc[mask, "metadata_from"]
        == df.loc[mask, "extracted_from"].str.split("<").str[0]
    )
    | (df.loc[mask, "extracted_from"] == "H")
) & (df.loc[mask, "extracted_release_in_part_or_full"] == "RELEASE IN FULL")
results["CONS"]["CONS_SEMAN"]["metadata_from#extracted_from"] = r2.sum() / mask.sum()

mask = valid_rows.metadata_date_sent & valid_rows.extracted_date_sent
c = "metadata_date_sent"
dt1 = pd.to_datetime(df.loc[valid_rows[c], c], errors="coerce")
c = "extracted_date_sent"
dt2 = pd.to_datetime(df.loc[valid_rows[c], c], errors="coerce")
r3 = (
    (dt1.dt.day == dt2.dt.day)
    & (dt1.dt.month == dt2.dt.month)
    & (dt1.dt.year == dt2.dt.year)
)
results["CONS"]["CONS_SEMAN"]["metadata_date_sent#extracted_date_sent"] = (
    r3.sum() / mask.sum()
)

mask = valid_rows.metadata_date_released & valid_rows.extracted_date_released
c = "metadata_date_released"
dt1 = pd.to_datetime(df.loc[valid_rows[c], c], errors="coerce")
c = "extracted_date_released"
dt2 = pd.to_datetime(df.loc[valid_rows[c], c], errors="coerce")
r4 = (
    (dt1.dt.day == dt2.dt.day)
    & (dt1.dt.month == dt2.dt.month)
    & (dt1.dt.year == dt2.dt.year)
)
results["CONS"]["CONS_SEMAN"]["metadata_date_released#extracted_date_released"] = (
    r4.sum() / mask.sum()
)

valid_rows.loc[r1.loc[~r1].index, "extracted_to"] = False
valid_rows.loc[r1.loc[~r1].index, "metadata_to"] = False
valid_rows.loc[
    (r1 & r2).loc[~(r1 & r2)].index, "extracted_release_in_part_or_full"
] = False
valid_rows.loc[r2.loc[~r2].index, "metadata_from"] = False
valid_rows.loc[r2.loc[~r2].index, "extracted_from"] = False
valid_rows.loc[r3.loc[~r3].index, "metadata_date_sent"] = False
valid_rows.loc[r3.loc[~r3].index, "extracted_date_sent"] = False
valid_rows.loc[r4.loc[~r4].index, "metadata_date_released"] = False
valid_rows.loc[r4.loc[~r4].index, "extracted_date_released"] = False

################################
# currentness (atualidade) CURR
################################

# CURR_UPD #####################

mask = valid_rows.metadata_date_sent & valid_rows.metadata_date_released
c = "metadata_date_sent"
dt1 = pd.to_datetime(df.loc[mask, c], errors="coerce")
c = "metadata_date_released"
dt2 = pd.to_datetime(df.loc[mask, c], errors="coerce")
resp = dt1.dt.year == dt2.dt.year
results["CURR"]["CURR_UPD"][c] = resp.sum() / mask.sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

mask = valid_rows.extracted_date_sent & valid_rows.extracted_date_released
c = "extracted_date_sent"
dt1 = pd.to_datetime(df.loc[mask, c], errors="coerce")
c = "extracted_date_released"
dt2 = pd.to_datetime(df.loc[mask, c], errors="coerce")
resp = dt1.dt.year == dt2.dt.year
results["CURR"]["CURR_UPD"][c] = resp.sum() / mask.sum()
valid_rows.loc[resp.loc[~resp].index, c] = False


################################
# uniqueness (unicidade) UNI
################################

# UNI_REG ######################
u = "person_id"
columns_uni = ["name", "doc_number", "alias"]
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
