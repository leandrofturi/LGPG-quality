# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


import pandas as pd


class Generalization:
    @staticmethod
    def anonymize(df, columns, rules):
        df_cpy = df.copy()
        for c in columns:
            if c in rules.keys():
                df_cpy[c] = df_cpy[c].astype(str)
                if rules[c]["type"] == "replace":
                    df_cpy[c] = df_cpy[c].str.replace(
                        rules[c]["string"], rules[c]["replaced"]
                    )
                elif rules[c]["type"] == "split":
                    df_cpy[c] = (
                        df_cpy[c].str.split(rules[c]["char"]).str[rules[c]["keep"]]
                    )
                elif rules[c]["type"] == "crop":
                    df_cpy[c] = df_cpy[c].str[rules[c]["start"], rules[c]["stop"]]
                elif rules[c]["type"] == "hist":
                    df_cpy[c] = pd.to_numeric(df_cpy[c], errors="coerce")
                    df_cpy[c] = pd.cut(
                        df_cpy[c],
                        rules[c]["nbins"],
                        labels=range(rules[c]["nbins"])
                    )
            else:
                df_cpy[c] = None
        return df_cpy
