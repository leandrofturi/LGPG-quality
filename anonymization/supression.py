# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


class Supression:
    @staticmethod
    def anonymize(df, columns):
        df_cpy = df.copy()
        df_cpy[columns] = None
        return df_cpy
