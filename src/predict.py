import click
import pandas as pd
from utils import predict_immunogenicity
from args import conf


@click.command()
@click.option(
    "--input-file",
    default="data/humab25_sequences.csv",
    help="Input CSV file containing antibody sequences.",
)
@click.option(
    "--output-file",
    default="predictions.csv",
    help="Output CSV file to save predictions.",
)
def predict(input_file, output_file):
    """
    Predicts immunogenicity for antibody sequences in a CSV file.
    """
    df = pd.read_csv(input_file)
    scores = []
    for index, row in df.iterrows():
        vh_seq = row["VH"]
        vl_seq = row["VL"]
        if pd.isna(vh_seq) or pd.isna(vl_seq):
            scores.append(None)
            continue

        full_seq = f"{vh_seq},{vl_seq}"
        score = predict_immunogenicity(full_seq, conf.min_mer, conf.max_mer)
        scores.append(score)
        print(f"Predicted immunogenicity for {row['name']}: {score}")

    df["immunogenicity_score"] = scores
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    predict()
