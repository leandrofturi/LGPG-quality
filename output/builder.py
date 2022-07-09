import json

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt


files = []
for d in [
    "athleteEvents",
    "Canada",
    "eleicoes",
    "EuropeanSoccer",
    "Poland",
    "Sinasc",
]:
    for m in [
        "raw",
        "supression",
        "generalization",
        "randomization",
        "pseudoanonymization",
    ]:
        files.append(d + "_" + m + ".json")


result = []
for f in files:
    with open(f"output/{f}") as json_file:
        data = json.load(json_file)
        result.append({f.split(".json")[0]: data})

rows = {
    list(r.keys())[0]: [
        list(r.values())[0].get("COMP"),
        list(r.values())[0].get("ACC"),
        list(r.values())[0].get("CRED"),
        list(r.values())[0].get("CONS"),
        list(r.values())[0].get("CURR"),
        list(r.values())[0].get("UNI"),
    ]
    for r in result
}
df = pd.DataFrame.from_dict(rows).T
df.columns = ["COMP", "ACC", "CRED", "CONS", "CURR", "UNI"]
df["dataset"], df["test"] = zip(*map(lambda x: x.split("_"), df.index))
df["result"] = df[["COMP", "ACC", "CRED", "CONS", "CURR", "UNI"]].prod(axis=1).abs()

std_ = df.groupby("dataset").apply(
    lambda x: (x["result"] - x.loc[x["test"] == "raw", "result"].iloc[0])
    / x.loc[x["test"] == "raw", "result"].iloc[0]
)
df = df.join(pd.DataFrame({"dev": std_.reset_index(level=[0])["result"]}))

sorter = ["raw", "supression", "generalization", "randomization", "pseudoanonymization"]
sorterIndex = dict(zip(sorter, range(len(sorter))))
df["test_rank"] = df["test"].map(sorterIndex)
df = df.sort_values(["dataset", "test_rank"])
del df["test_rank"]

df = df.set_index(["dataset", "test"])
df.to_excel("output/results.xlsx")


cmap = matplotlib.cm.get_cmap("viridis", 5)
for m in ["COMP", "ACC", "CRED", "CONS", "CURR", "UNI"]:
    v = df[[m]].reset_index()
    v = v.loc[v[m] > 0, :]
    v.loc[v.test == "raw", "test"] = "Original"
    v.loc[v.test == "supression", "test"] = "SUP"
    v.loc[v.test == "generalization", "test"] = "GEN"
    v.loc[v.test == "randomization", "test"] = "RAND"
    v.loc[v.test == "pseudoanonymization", "test"] = "PSAN"
    plt.figure()
    for i, d in enumerate(
        [
            "athleteEvents",
            "Canada",
            "eleicoes",
            "EuropeanSoccer",
            "Poland",
            "Sinasc",
        ]
    ):
        w = v.loc[v["dataset"] == d, :]
        plt.bar(
            w["test"],
            w[m],
            0.5,
            label=d,
            align="center",
            color=matplotlib.colors.rgb2hex(cmap(i)),
        )

    v.pivot(*v)[["Original", "SUP", "RAND", "GEN", "PSAN"]].plot(
        kind="bar", cmap="viridis"
    )
    plt.xticks(rotation=45)
    plt.xlabel("Base de dados")
    plt.ylabel(m)
    plt.legend(
        frameon=True,
        framealpha=1,
        edgecolor="black",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=5,
    )
    plt.tight_layout()
    plt.savefig(f"output/{m}.png")
