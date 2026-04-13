"""
Command-line entry point for the data injection pipeline.

Usage (inside the container or via docker compose run):

    # Full pipeline — all tickers, inject into DB
    python scripts/pipeline.py run

    # With tuning params
    python scripts/pipeline.py run --procs 8 --concur 100

    # Random sample of N tickers
    python scripts/pipeline.py sample --n 50 --procs 4 --concur 100
"""
import argparse
import asyncio
import logging


def _setup_logging() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full pipeline for all tickers and inject results into the DB."""
    from app.utils.pipeline import run_pipeline_multiprocess, save_to_db

    results, _ = run_pipeline_multiprocess(
        num_processes=args.procs,
        max_concurrent=args.concur,
    )
    asyncio.run(save_to_db(results))


def cmd_sample(args: argparse.Namespace) -> None:
    """Run the pipeline on a random sample of N tickers and inject into the DB."""
    from app.utils.pipeline import run_sample

    run_sample(
        n=args.n,
        num_processes=args.procs,
        max_concurrent=args.concur,
        save_db=True,
    )


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Data injection pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run: full pipeline ────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Run the full pipeline for ALL tickers and inject into the DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_run.add_argument("--procs",  type=int, default=8,   help="Number of worker processes")
    p_run.add_argument("--concur", type=int, default=100, help="Concurrent HTTP requests per process")

    # ── sample: random subset ─────────────────────────────────────────────────
    p_sample = sub.add_parser(
        "sample",
        help="Run the pipeline on N randomly selected tickers and inject into the DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_sample.add_argument("--n",     type=int, default=50,  help="Number of random tickers to sample")
    p_sample.add_argument("--procs",  type=int, default=8,   help="Number of worker processes")
    p_sample.add_argument("--concur", type=int, default=100, help="Concurrent HTTP requests per process")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "sample":
        cmd_sample(args)


if __name__ == "__main__":
    main()
