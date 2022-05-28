# Métricas de Qualidade de Dados adequadas à Lei Geral de Proteção de Dados Pessoais
# @leandrofturi


class Generalization:
    @staticmethod
    def anonymize(df, columns, rules):
        for c in columns:
            df[c] = df[c].astype(str)
            if rules[c]["type"] == "replace":
                df[c].str.replace(rules[c]["string"], rules[c]["replaced"])
            if rules[c]["type"] == "split":
                df[c].str.split(rules[c]["char"]).str[rules[c]["keep"]]
            if rules[c]["type"] == "crop":
                df[c].str[rules[c]["start"], rules[c]["stop"]]
        return df
