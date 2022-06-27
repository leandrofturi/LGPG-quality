import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# from cleaner.Sinasc import cleaner
from anonymization.supression import Supression
from anonymization.randomization import Randomization
from anonymization.generalization import Generalization
from anonymization.pseudoanonymization import PseudoAnonymization

from cleaner.Sinasc import LGPD_COLUMNS, rules

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
# cross_validate(clf, X, y, cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42), scoring='accuracy')
# clf['estimator'].best_params_
# {'bootstrap': True, 'class_weight': 'balanced_subsample', 'max_features': None, 'n_estimators': 100}

df = pd.read_parquet("datasets/Sinasc.parquet")
valid_rows = cleaner(df)
v = valid_rows.sum() / len(df.index)
v_cols = v.loc[v >= 0.85]

# ['loc_nasc', 'idade_mae', 'est_civ_mae', 'esc_mae', 'qtd_fil_vivo',
#  'qtd_fil_mort', 'gestacao', 'gravidez', 'parto', 'consultas', 'sexo',
#  'apgar_1', 'apgar_5', 'raca_cor', 'peso', 'cod_estab']


def learn(df, out_filename):
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
        ]
    ].iloc[:30000]
    y = df[["qtd_fil_mort", "qtd_fil_vivo"]].iloc[:30000]
    y = (y.qtd_fil_vivo > y.qtd_fil_mort).astype(int)

    enc = {}
    for c in X.select_dtypes(include=["string", "object", "category"]).columns:
        enc[c] = LabelEncoder()
        X.loc[X.index, c] = enc[c].fit_transform(X[c].astype(str))
    X.fillna(-1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestClassifier(
        bootstrap=True, class_weight="balanced", max_features="sqrt", n_estimators=100
    )
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    final = {
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "accuracy_score": accuracy_score(y_test, predictions),
    }

    with open(out_filename, "w") as f:
        json.dump(final, f, indent=4, sort_keys=False)


################################
# learn
################################

df = pd.read_parquet("datasets/Sinasc.parquet")
learn(df, K, "output/rf_Sinasc_raw.png")
learn(Supression.anonymize(df, LGPD_COLUMNS), K, "output/rf_Sinasc_supression.png")
learn(
    Randomization.anonymize(df, LGPD_COLUMNS), K, "output/rf_Sinasc_randomization.png"
)
learn(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    K,
    "output/rf_Sinasc_generalization.png",
)
learn(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/rf_Sinasc_pseudoanonymization.png",
)
