import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score, precision_recall_fscore_support

# from cleaner.Sinasc import cleaner
from anonymization.supression import Supression
from anonymization.randomization import Randomization
from anonymization.generalization import Generalization
from anonymization.pseudoanonymization import PseudoAnonymization

LGPD_COLUMNS = [
    "loc_nasc",
    "cod_mun_nasc",
    "idade_mae",
    "est_civ_mae",
    "esc_mae",
    "qtd_fil_vivo",
    "qtd_fil_mort",
    "cod_mun_res",
    "gestacao",
    "gravidez",
    "parto",
    "consultas",
    "dt_nasc",
    "sexo",
    "apgar_1",
    "apgar_5",
    "raca_cor",
    "peso",
    "cod_anomal",
    "cod_estab",
    "cod_ocup_mae",
    "id_anomal",
    "cod_bai_nasc",
    "cod_bai_res",
    "uf_inform",
    "hora_nasc",
    "dt_cadastro",
    "dt_recebim",
    "origem",
    "cod_cart",
    "num_reg_cart",
    "dt_reg_cart",
    "cod_pais_res",
    "numero_lote",
    "versao_sist",
    "dif_data",
    "dt_rec_orig",
    "natural_mae",
    "cod_mun_natu",
    "seri_esc_mae",
    "dt_nasc_mae",
    "raca_cor_mae",
    "qtd_gest_ant",
    "qtd_part_nor",
    "qtd_part_ces",
    "idade_pai",
    "dt_ult_menst",
    "sema_gestac",
    "tp_met_estim",
    "cons_prenat",
    "mes_prenat",
    "tp_apresent",
    "st_trab_part",
    "st_ces_parto",
    "tp_robson",
    "std_nepidem",
    "std_nova",
    "raca_cor_rn",
    "raca_cor_n",
    "esc_mae_2010",
    "cod_mun_cart",
    "cod_uf_natu",
    "tp_nasc_assi",
    "esc_mae_agr_1",
    "dt_rec_orig_a",
    "tp_func_resp",
    "td_doc_resp",
    "dt_declarac",
    "par_idade",
    "kotelchuck",
]

rules = {
    "cod_mun_nasc": {"type": "crop", "start": 0, "stop": 2},
    "idade_mae": {"type": "hist", "nbins": 5},
    "qtd_fil_vivo": {"type": "hist", "nbins": 3},
    "qtd_fil_mort": {"type": "hist", "nbins": 3},
    "cod_mun_res": {"type": "crop", "start": 0, "stop": 2},
    "cod_mun_natu": {"type": "crop", "start": 0, "stop": 2},
    "qtd_gest_ant": {"type": "hist", "nbins": 3},
    "qtd_part_nor": {"type": "hist", "nbins": 3},
    "qtd_part_ces": {"type": "hist", "nbins": 3},
    "idade_pai": {"type": "hist", "nbins": 5},
    "sema_gestac": {"type": "hist", "nbins": 5},
    "cons_prenat": {"type": "hist", "nbins": 3},
}

# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_validate, GridSearchCV, RepeatedStratifiedKFold

# params = {
#     'n_estimators': [10, 20, 50, 100],
#     'max_features': [None, 'sqrt', 'log2'],
#     'bootstrap': [True, False],
#     'class_weight': [None, 'balanced', 'balanced_subsample']
# }

# clf = Pipeline([('estimator', GridSearchCV(RandomForestClassifier(), param_grid=params, scoring='accuracy',
#                 cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=42)))]).fit(X, y) # Grid search
# clf['estimator'].best_params_
# {'bootstrap': True, 'class_weight': None, 'max_features': 'sqrt', 'n_estimators': 100}

# df = pd.read_parquet("datasets/Sinasc.parquet")
# valid_rows = cleaner(df)
# v = valid_rows.sum() / len(df.index)
# v_cols = v.loc[v >= 0.85]

# ['loc_nasc', 'idade_mae', 'est_civ_mae', 'esc_mae', 'qtd_fil_vivo',
#  'qtd_fil_mort', 'gestacao', 'gravidez', 'parto', 'consultas', 'sexo',
#  'apgar_1', 'apgar_5', 'raca_cor', 'peso', 'cod_estab']


def learn(df, y, out_filename):
    X = df[
        [
            "loc_nasc",
            "idade_mae",
            "est_civ_mae",
            "esc_mae",
            "gestacao",
            "gravidez",
            "parto",
            "consultas",
            "sexo",
            "apgar_1",
            "apgar_5",
            "raca_cor",
            "peso",
            "cod_estab",
            "qtd_fil_vivo",
        ]
    ].iloc[:30000]

    enc = {}
    for c in X.select_dtypes(include=["string", "object", "category"]).columns:
        enc[c] = LabelEncoder()
        X.loc[X.index, c] = enc[c].fit_transform(X[c].astype(str)).astype(int)
    X.fillna(-1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestClassifier(
        bootstrap=True, class_weight=None, max_features="sqrt", n_estimators=100
    )
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    plt.figure()
    plot_confusion_matrix(
        rf,
        X_test,
        y_test,
        cmap="viridis",
        normalize="all",
        display_labels=["F", "V"],
    )
    plt.xlabel('Classes previstas')
    plt.ylabel('Classes verdadeiras')
    plt.tight_layout()
    plt.savefig(f"{out_filename.replace('rf', 'rf_cf').replace('.json', '.png')}")

    metrics = precision_recall_fscore_support(y_test, predictions)
    final = {
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "accuracy_score": accuracy_score(y_test, predictions),
        "adjusted_rand_score": adjusted_rand_score(y_test, predictions),
        "precision": metrics[0].tolist(),
        "recall": metrics[1].tolist(),
        "fscore": metrics[2].tolist(),
        "support": metrics[3].tolist(),
    }

    with open(out_filename, "w") as f:
        json.dump(final, f, indent=4, sort_keys=False)


################################
# learn
################################

df = pd.read_parquet("datasets/Sinasc.parquet")
y = df["qtd_fil_mort"].iloc[:30000]
y = (y != 0).astype(int)

learn(df, y, "output/rf_Sinasc_raw.json")
learn(Supression.anonymize(df, LGPD_COLUMNS), y, "output/rf_Sinasc_supression.json")
learn(
    Randomization.anonymize(df, LGPD_COLUMNS), y, "output/rf_Sinasc_randomization.json"
)
learn(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    y,
    "output/rf_Sinasc_generalization.json",
)
learn(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    y,
    "output/rf_Sinasc_pseudoanonymization.json",
)
