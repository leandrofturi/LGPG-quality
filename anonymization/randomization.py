# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


class Randomization:
    @staticmethod
    def anonymize(df, columns):
        df_cpy = df.copy()
        for c in columns:
            df_cpy[c] = df_cpy[c].sample(frac=1, random_state=42).reset_index(drop=True)
        return df_cpy
