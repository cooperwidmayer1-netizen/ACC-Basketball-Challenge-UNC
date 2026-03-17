import argparse

from src.acc_model.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ACC Basketball Challenge pipeline.")
    parser.add_argument(
        "--csv-schedule",
        required=True,
        help="Path to the authority CSV schedule file.",
    )
    parser.add_argument(
        "--out",
        default="outputs/predictions.csv",
        help="Where to save future predictions.",
    )
    parser.add_argument(
        "--tuning-out",
        default="outputs/tuning.csv",
        help="Where to save the tuning results table.",
    )

    args = parser.parse_args()

    run(
        csv_schedule=args.csv_schedule,
        out_pred=args.out,
        out_tuning=args.tuning_out,
    )


if __name__ == "__main__":
    main()
