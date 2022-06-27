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


################################
# run
################################

K = 10

df = pd.read_parquet("datasets/Sinasc.parquet")
learn(df, K, "output/Sinasc_raw.png")
learn(Supression.anonymize(df, LGPD_COLUMNS), K, "output/Sinasc_supression.png")
learn(Randomization.anonymize(df, LGPD_COLUMNS), K, "output/Sinasc_randomization.png")
learn(
    Generalization.anonymize(df, LGPD_COLUMNS, rules),
    K,
    "output/Sinasc_generalization.png",
)
learn(
    PseudoAnonymization.anonymize(df, LGPD_COLUMNS),
    K,
    "output/Sinasc_pseudoanonymization.png",
)
