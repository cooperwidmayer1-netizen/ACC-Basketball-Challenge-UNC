import argparse
from src.acc_model.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-schedule", required=True)
    parser.add_argument("--out", default="outputs/predictions.csv")
    parser.add_argument("--tuning-out", default="outputs/tuning.csv")

    args = parser.parse_args()

    run(
        csv_schedule=args.csv_schedule,
        out_pred=args.out,
        out_tuning=args.tuning_out
    )
