import base64

datasets = [
    "athleteEvents",
    "Canada",
    "eleicoes",
    "EuropeanSoccerDatabase",
    "Poland",
    "Sinasc",
]


with open(f"output/kmeans.html", "w") as out_file:
    for dataset in datasets:
        out_file.write(f"<h1>{dataset}</h1>\n")

        enc = base64.b64encode(open(f"output/elbow_{dataset}.png", "rb").read()).decode(
            "ascii"
        )
        enc2 = base64.b64encode(
            open(f"output/scatter_{dataset}_raw.png", "rb").read()
        ).decode("ascii")
        enc3 = base64.b64encode(
            open(f"output/scatter_{dataset}_supression.png", "rb").read()
        ).decode("ascii")
        enc4 = base64.b64encode(
            open(f"output/scatter_{dataset}_generalization.png", "rb").read()
        ).decode("ascii")
        enc5 = base64.b64encode(
            open(f"output/scatter_{dataset}_randomization.png", "rb").read()
        ).decode("ascii")
        enc6 = base64.b64encode(
            open(f"output/scatter_{dataset}_pseudoanonymization.png", "rb").read()
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
            open(f"output/feature_weights_{dataset}_raw.png", "rb").read()
        ).decode("ascii")
        enc3 = base64.b64encode(
            open(f"output/feature_weights_{dataset}_supression.png", "rb").read()
        ).decode("ascii")
        enc4 = base64.b64encode(
            open(f"output/feature_weights_{dataset}_generalization.png", "rb").read()
        ).decode("ascii")
        enc5 = base64.b64encode(
            open(f"output/feature_weights_{dataset}_randomization.png", "rb").read()
        ).decode("ascii")
        enc6 = base64.b64encode(
            open(
                f"output/feature_weights_{dataset}_pseudoanonymization.png", "rb"
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
