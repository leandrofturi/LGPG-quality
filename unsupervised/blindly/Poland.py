import pandas as pd

from unsupervised.model import learn
from anonymization.supression import Supression
from anonymization.randomization import Randomization
from anonymization.generalization import Generalization
from anonymization.pseudoanonymization import PseudoAnonymization


################################
# LGPD columns
################################

LGPD_COLUMNS = [
    "date_and_place_of_birth",
    "occupation",
    "education",
    "won_in_the_elections",
    "election_committee",
    "name",
    "club_circle",
    "district",
]

rules = {
    "date_and_place_of_birth": {"type": "split", "char": " ", "keep": -1},
    "occupation": {"type": "split", "char": " ", "keep": 0},
    "name": {"type": "split", "char": " ", "keep": 0},
}

################################
# run
################################

K = 9

df = pd.read_parquet("datasets/Poland.parquet")
learn(df, K, "output/Poland_raw.png")
learn(Supression.anonymize(df, LGPD_COLUMNS), K, "output/Poland_supression.png")
learn(Randomization.anonymize(df, LGPD_COLUMNS), K, "output/Poland_randomization.png")
learn(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    K,
    "output/Poland_generalization.png",
)
learn(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/Poland_pseudoanonymization.png",
)
