# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import numpy as np
import pandas as pd

################################
# load data
################################

df1 = pd.read_parquet("datasets/eleicoes1.parquet")
df2 = pd.read_parquet("datasets/eleicoes2.parquet")
df3 = pd.read_parquet("datasets/eleicoes3.parquet")
df = df1.append(df2).append(df3)
valid_rows = pd.DataFrame(True, index=df.index, columns=df.columns)
results = {
    "COMP": {"COMP_REG": {}},
    "ACC": {"ACC_SINT": {}, "RAN_ACC": {}, "ACC_SEMAN": {}},
    "CRED": {"CRED_VAL_DAT": {}},
    "CONS": {"CONS_SEMAN": {}},
    "CURR": {"CURR_UPD": {}},
    "UNI": {"UNI_REG": {}},
}

LGPD_COLUMNS = ["etnia", "partido", "sigla_partido"]


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

DICT = {
    "declara_bens": ["S", "N", "NAO DIVULGAVEL"],
    "cargo": [
        "VEREADOR",
        "PREFEITO",
        "VICE-PREFEITO",
        "DEPUTADO ESTADUAL",
        "DEPUTADO FEDERAL",
        "DEPUTADO DISTRITAL",
        "2o SUPLENTE SENADOR",
        "1o SUPLENTE SENADOR",
        "SENADOR",
        "VICE-GOVERNADOR",
        "GOVERNADOR",
        "PRESIDENTE",
        "VICE-PRESIDENTE",
    ],
    "estado_civil": [
        "CASADO(A)",
        "SOLTEIRO(A)",
        "DIVORCIADO(A)",
        "SEPARADO(A) JUDICIALMENTE",
        "VIUVO(A)",
        "NAO INFORMADO",
        "NAO DIVULGAVEL",
    ],
    "genero": ["MASCULINO", "FEMININO", "NAO DIVULGAVEL", "NAO INFORMADO"],
    "grau_instrucao": [
        "ENSINO MEDIO COMPLETO",
        "SUPERIOR COMPLETO",
        "ENSINO FUNDAMENTAL INCOMPLETO",
        "ENSINO FUNDAMENTAL COMPLETO",
        "SUPERIOR INCOMPLETO",
        "MEDIO COMPLETO",
        "ENSINO MEDIO INCOMPLETO",
        "LE E ESCREVE",
        "1O GRAU INCOMPLETO",
        "FUNDAMENTAL INCOMPLETO",
        "2O GRAU COMPLETO",
        "1O GRAU COMPLETO",
        "FUNDAMENTAL COMPLETO",
        "2O GRAU INCOMPLETO",
        "MEDIO INCOMPLETO",
        "NAO INFORMADO",
        "NAO DIVULGAVEL",
        "ANALFABETO",
    ],
    "etnia": [
        "BRANCA",
        "PARDA",
        "PRETA",
        "NAO INFORMADO",
        "AMARELA",
        "INDIGENA",
        "NAO DIVULGAVEL",
    ],
}
for c in DICT.keys():
    resp = df.loc[valid_rows[c], c].isin(DICT[c])
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

c = "ano"
values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y", errors="coerce")
resp = ~values.isna()
results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

c = "data_nascimento"
values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce")
resp = ~values.isna()
results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

DICT_SIZES = {"cpf": 11, "titulo_eleitoral": 12}

for c in DICT_SIZES.keys():
    resp = df.loc[valid_rows[c], c].astype(str).str.len() == DICT_SIZES[c]
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

mun_ibge = pd.read_csv("utils/municipiosIBGE.csv")

columns_mun_ibge = ["sigla_unidade_federativa", "sigla_unidade_federativa_nascimento"]
for c in columns_mun_ibge:
    values = df.loc[valid_rows[c], c]
    resp = values.isin(mun_ibge["UF"])
    results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
    valid_rows.loc[resp.loc[~resp].index, c] = False

c = "municipio_nascimento"
values = df.loc[valid_rows[c], c]
resp = values.str.upper().isin(mun_ibge["nome"].str.upper())
results["ACC"]["ACC_SINT"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

# RAN_ACC ######################

min_date = pd.to_datetime("1996-01-01")
max_date = pd.to_datetime("2020-12-31")

c = "ano"
values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y", errors="coerce")
resp = (values >= min_date) & (values <= max_date)
results["ACC"]["RAN_ACC"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

c = "data_nascimento"
values = pd.to_datetime(df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce")
resp = values <= max_date
results["ACC"]["RAN_ACC"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False


# ACC_SEMAN ####################

c = "data_nascimento"
values = pd.to_datetime(
    df.loc[valid_rows[c], c], format="%Y-%m-%d", errors="coerce"
).dt.to_pydatetime()
resp = (pd.Series(max_date - values).apply(lambda x: x.days) / 365) <= 120
results["ACC"]["ACC_SEMAN"][c] = resp.sum() / valid_rows[c].sum()
valid_rows.loc[resp.loc[~resp].index, c] = False

pleitos = pd.read_csv("utils/eleicoes.csv")

c = "ano"
resp = df.loc[valid_rows[c], c].isin(pleitos["pleito_ano"])
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
mask = valid_rows.sigla_unidade_federativa_nascimento & valid_rows.municipio_nascimento
r1 = (
    df.loc[mask, :]
    .groupby("sigla_unidade_federativa_nascimento")
    .apply(
        lambda x: x["municipio_nascimento"]
        .str.upper()
        .isin(
            mun_ibge.loc[
                mun_ibge["UF"] == x["sigla_unidade_federativa_nascimento"].iloc[0],
                "nome",
            ].str.upper()
        )
    )
)
results["CONS"]["CONS_SEMAN"][
    "sigla_unidade_federativa_nascimento#municipio_nascimento"
] = (r1.sum() / mask.sum())

mask = valid_rows.cpf
r2 = ~df.loc[mask, "cpf"].isnull()
results["CONS"]["CONS_SEMAN"]["cpf"] = r2.sum() / mask.sum()

mask = valid_rows.titulo_eleitoral
r3 = ~df.loc[mask, "titulo_eleitoral"].isnull()
results["CONS"]["CONS_SEMAN"]["titulo_eleitoral"] = r3.sum() / mask.sum()

mask = valid_rows.despesa_maxima_campanha
r4 = df.loc[mask, "despesa_maxima_campanha"] >= 0
results["CONS"]["CONS_SEMAN"]["despesa_maxima_campanha"] = r4.sum() / mask.sum()


################################
# currentness (atualidade) CURR
################################

# CURR_UPD #####################


################################
# uniqueness (unicidade) UNI
################################

# UNI_REG ######################

u = "cpf"
columns_uni = [
    "cpf",
    "data_nascimento",
    "etnia",
    "nacionalidade",
    "nome",
    "municipio_nascimento",
    "sigla_unidade_federativa_nascimento",
    "titulo_eleitoral",
]
for c in columns_uni:
    mask = valid_rows[c] & valid_rows[u]
    r = []
    for i in df[u].unique():
        values = df.loc[(df[u] == i) & mask, c]
        if len(values) > 1:
            resp = values.ne(values.shift().bfill())
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
