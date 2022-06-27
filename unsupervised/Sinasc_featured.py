import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster.elbow import kelbow_visualizer

from cleaner.Sinasc import cleaner
from unsupervised.model_featured import learn
from anonymization.supression import Supression
from anonymization.randomization import Randomization
from anonymization.generalization import Generalization
from anonymization.pseudoanonymization import PseudoAnonymization

from cleaner.Sinasc import LGPD_COLUMNS, rules

np.random.seed(42)


################################
# encoding
################################

# numeric
"qtd_fil_vivo", "qtd_fil_mort", "apgar_1", "apgar_5", "peso",
"qtd_gest_ant", "qtd_part_nor", "qtd_part_ces", "idade_pai"
"sema_gestac", "cons_prenat", "mes_prenat", "tp_robson",
"kotelchuck"

# sorted categorical
"cod_mun_nasc", "cod_mun_res", "gestacao", "gravidez", "consultas"
"cod_bai_nasc", "cod_bai_res", "uf_inform", "cod_pais_res", "cod_mun_natu"
"natural_mae", "seri_esc_mae", "esc_mae", "esc_mae_2010",
"cod_mun_cart", "cod_uf_natu", "esc_mae_agr_1", "cod_anomal"

# unsorted categorical
"cod_ocup_mae",

# small domain categorical
"loc_nasc", "est_civ_mae", "id_anomal", "parto", "sexo",
"raca_cor", "origem", "raca_cor_mae", "tp_met_estim",
"tp_apresent", "st_trab_part", "st_ces_parto", "std_nepidem",
"std_nova", "raca_cor_rn", "raca_cor_n", "tp_nasc_assi",
"tp_func_resp", "td_doc_resp", "par_idade"

# dates
"dt_nasc", "dt_cadastro", "dt_recebim", "dt_reg_cart", "dt_rec_orig",
"dt_rec_orig_a", "dt_nasc_mae", "hora_nasc", "dt_ult_menst",
"dt_declarac", "dif_data",

# remove
"numero_lote", "versao_sist", "cod_estab", "cod_cart", "num_reg_cart"

df = pd.read_parquet("datasets/Sinasc.parquet")
valid_rows = cleaner(df)
v = valid_rows.sum() / len(df.index)
v_cols = v.loc[v >= 0.85]


def categorize(df):
    # pass
    X = df[
        [
            "idade_mae",
            "esc_mae",
            "qtd_fil_vivo",
            "gestacao",
            "gravidez",
            "consultas",
            "apgar_1",
            "apgar_5",
            "peso",
        ]
    ].copy()

    X["hora_nasc"] = df["hora_nasc"].apply(lambda x: re.findall(r"\d+", str(x)))
    X["hora_nasc"] = X["hora_nasc"].apply(lambda x: float("".join(x)) if x else np.nan)

    for c in ["loc_nasc", "est_civ_mae", "parto", "sexo", "raca_cor", "id_anomal"]:
        X = pd.concat([X, pd.get_dummies(df[c].astype(str), prefix=c)], axis=1)

    X["qtd_fil_mort"] = df["qtd_fil_mort"]

    X.fillna(-1, inplace=True)
    X_scl = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    return X_scl


################################
# elbow
################################

df = pd.read_parquet("datasets/Sinasc.parquet")
X = categorize(df)

kelbow_visualizer(KMeans(), X, k=(2, 24), title="Sinasc")

K = 21

################################
# learn
################################

df = pd.read_parquet("datasets/Sinasc.parquet")
learn(categorize(df)[:30000], K, "output/km_Sinasc_raw.png")
learn(
    categorize(Supression.anonymize(df, LGPD_COLUMNS))[:30000],
    K,
    "output/km_Sinasc_supression.png",
)
learn(
    categorize(Randomization.anonymize(df, LGPD_COLUMNS))[:30000],
    K,
    "output/km_Sinasc_randomization.png",
)
learn(
    categorize(Generalization.anonymize(df, LGPD_COLUMNS, rules))[:30000],
    K,
    "output/km_Sinasc_generalization.png",
)
learn(
    categorize(PseudoAnonymization.anonymize(df, LGPD_COLUMNS))[:30000],
    K,
    "output/km_Sinasc_pseudoanonymization.png",
)
