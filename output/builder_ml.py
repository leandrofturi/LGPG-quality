import base64
import json

datasets = [
    "eleicoes",
    "Sinasc",
]


with open(f"output/kmeans.html", "w") as out_file:
    for dataset in datasets:
        out_file.write(f"<h1>{dataset}</h1>\n")

        enc = base64.b64encode(
            open(f"output/elbow_{dataset}_featured.png", "rb").read()
        ).decode("ascii")
        enc2 = base64.b64encode(
            open(f"output/scatter_km_{dataset}_raw.png", "rb").read()
        ).decode("ascii")
        enc3 = base64.b64encode(
            open(f"output/scatter_km_{dataset}_supression.png", "rb").read()
        ).decode("ascii")
        enc4 = base64.b64encode(
            open(f"output/scatter_km_{dataset}_generalization.png", "rb").read()
        ).decode("ascii")
        enc5 = base64.b64encode(
            open(f"output/scatter_km_{dataset}_randomization.png", "rb").read()
        ).decode("ascii")
        enc6 = base64.b64encode(
            open(f"output/scatter_km_{dataset}_pseudoanonymization.png", "rb").read()
        ).decode("ascii")

        out_file.write(f"<img src='data:image/png;base64,{enc}' width='50%'>\n")
        out_file.write(f"<img src='data:image/png;base64,{enc2}' width='50%'>\n")
        out_file.write('<p float="middle">')
        out_file.write(f"<img src='data:image/png;base64,{enc3}' width='50%'>")
        out_file.write(f"<img src='data:image/png;base64,{enc4}' width='50%'>")
        out_file.write("</p>\n")
        out_file.write('<p float="middle">')
        out_file.write(f"<img src='data:image/png;base64,{enc5}' width='50%'>")
        out_file.write(f"<img src='data:image/png;base64,{enc6}'  width='50%'>")
        out_file.write("</p>\n")

        enc2 = base64.b64encode(
            open(f"output/feature_weights_km_{dataset}_raw.png", "rb").read()
        ).decode("ascii")
        enc3 = base64.b64encode(
            open(f"output/feature_weights_km_{dataset}_supression.png", "rb").read()
        ).decode("ascii")
        enc4 = base64.b64encode(
            open(f"output/feature_weights_km_{dataset}_generalization.png", "rb").read()
        ).decode("ascii")
        enc5 = base64.b64encode(
            open(f"output/feature_weights_km_{dataset}_randomization.png", "rb").read()
        ).decode("ascii")
        enc6 = base64.b64encode(
            open(
                f"output/feature_weights_km_{dataset}_pseudoanonymization.png", "rb"
            ).read()
        ).decode("ascii")

        out_file.write(f"<img src='data:image/png;base64,{enc2}' width='50%'>\n")
        out_file.write('<p float="middle">')
        out_file.write(f"<img src='data:image/png;base64,{enc3}' width='50%'>")
        out_file.write(f"<img src='data:image/png;base64,{enc4}' width='50%'>")
        out_file.write("</p>\n")
        out_file.write('<p float="middle">')
        out_file.write(f"<img src='data:image/png;base64,{enc5}' width='50%'>")
        out_file.write(f"<img src='data:image/png;base64,{enc6}'  width='50%'>")
        out_file.write("</p>\n")

        out_file.write("\n<hr />\n\n")


with open(f"output/rf.html", "w") as out_file:
    for dataset in datasets:
        out_file.write(f"<h1>{dataset}</h1>\n")

        with open(f"output/rf_{dataset}_raw.json", "r") as f:
            data = json.load(f)
        with open(f"output/rf_{dataset}_supression.json", "r") as f:
            data1 = json.load(f)
        with open(f"output/rf_{dataset}_generalization.json", "r") as f:
            data2 = json.load(f)
        with open(f"output/rf_{dataset}_randomization.json", "r") as f:
            data3 = json.load(f)
        with open(f"output/rf_{dataset}_pseudoanonymization.json", "r") as f:
            data4 = json.load(f)

        for d, t in zip(
            [data, data1, data2, data3, data4],
            [
                "raw",
                "supression",
                "generalization",
                "randomization",
                "pseudoanonymization",
            ],
        ):
            out_file.write(f"<h2>{t}</h2>\n")
            out_file.write(f'<p>accuracy_score={d["accuracy_score"]}</p>\n')
            out_file.write(f'<p>rand_score={d["rand_score"]}</p>\n')
            out_file.write("<table>")
            out_file.write("<tr>")
            out_file.write(f'<td>{d["confusion_matrix"][0][0]}</td>')
            out_file.write(f'<td>{d["confusion_matrix"][0][1]}</td>')
            out_file.write("</tr>")

            out_file.write("<tr>")
            out_file.write(f'<td>{d["confusion_matrix"][1][0]}</td>')
            out_file.write(f'<td>{d["confusion_matrix"][1][1]}</td>')
            out_file.write("</tr>")

            out_file.write("</table>")
