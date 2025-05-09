"""
Generate evaluations for offline test data.

For an evaluation run NAME, it reads outputs/NAME-recommendation.parquet and
produces OUTPUTS/name-profile-eval-metrics.csv.gz and OUTPUTS/name-metrics.json.

Usage:
    poprox_recommender.evaluation.evaluate [options] <name>

Options:
    -v, --verbose       enable verbose diagnostic logs
    --log-file=FILE     write log messages to FILE
    -M DATA, --mind-data=DATA
            read MIND test data DATA [default: MINDsmall_dev]
    -P DATA, --poprox-data=DATA
            read POPROX test data DATA
    -j N, --jobs=N
            use N parallel jobs
    -a --measure_all    evaluate all profiles including those with no clicks
    <name>              the name of the evaluation to measure
"""

# pyright: basic
import logging
import os
from typing import Any, Iterator
from uuid import UUID

import ipyparallel as ipp
import pandas as pd
from docopt import docopt
from lenskit.logging import LoggingConfig, item_progress

from poprox_recommender.config import available_cpu_parallelism
from poprox_recommender.data.eval import EvalData
from poprox_recommender.data.mind import MindData
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.evaluation.metrics import ProfileRecs, measure_profile_recs
from poprox_recommender.paths import project_root

logger = logging.getLogger("poprox_recommender.evaluation.evaluate")


def rec_profiles(eval_data: EvalData, profile_recs: pd.DataFrame, subset_truth=True) -> Iterator[ProfileRecs]:
    """
    Iterate over rec profiles, yielding each recommendation list with its truth and
    whether the profile is personalized.  This supports parallel computation of the
    final metrics.
    """
    for profile_id, recs in profile_recs.groupby("profile_id"):
        profile_id = UUID(str(profile_id))
        truth = eval_data.profile_truth(profile_id)
        assert truth is not None
        if subset_truth and len(truth) > 0:
            yield ProfileRecs(profile_id, recs.copy(), truth)
        elif not subset_truth:
            yield ProfileRecs(profile_id, recs.copy(), truth)


def profile_eval_results(
    eval_data: EvalData, profile_recs: pd.DataFrame, n_procs: int, subset_truth: bool = True
) -> Iterator[list[dict[str, Any]]]:
    if n_procs > 1:
        logger.info("starting parallel measurement with %d workers", n_procs)
        with ipp.Cluster(n=n_procs) as client:
            lb = client.load_balanced_view()
            yield from lb.imap(
                measure_profile_recs,
                rec_profiles(eval_data, profile_recs, subset_truth),
                ordered=False,
                max_outstanding=n_procs * 10,
            )
    else:
        for profile in rec_profiles(eval_data, profile_recs, subset_truth):
            yield measure_profile_recs(profile)


def main():
    options = docopt(__doc__)  # type: ignore
    log_cfg = LoggingConfig()
    if options["--verbose"] or os.environ.get("RUNNER_DEBUG", 0):
        log_cfg.set_verbose(True)
    if options["--measure_all"]:
        subset_truth = False
    else:
        subset_truth = True
    if options["--log-file"]:
        log_cfg.set_log_file(options["--log-file"])
    log_cfg.apply()

    global eval_data

    if options["--poprox-data"]:
        eval_data = PoproxData(options["--poprox-data"])
    else:
        eval_data = MindData(options["--mind-data"])

    n_jobs = options["--jobs"]
    if n_jobs is not None:
        n_jobs = int(n_jobs)
        if n_jobs <= 0:
            logger.warning("--jobs must be positive, using single job")
            n_jobs = 1
    else:
        n_jobs = available_cpu_parallelism(4)

    eval_name = options["<name>"]
    logger.info("measuring evaluation %s", eval_name)
    recs_fn = project_root() / "outputs" / eval_name / "recommendations"
    logger.info("loading recommendations from %s", recs_fn)
    recs_df = pd.read_parquet(recs_fn)
    n_profiles = recs_df["profile_id"].nunique()
    logger.info("loaded recommendations for %d profiles", n_profiles)

    logger.info("measuring recommendations")
    records = []
    with (
        item_progress("evaluate", total=n_profiles) as pb,
    ):
        for profile_rows in profile_eval_results(eval_data, recs_df, n_jobs, subset_truth):
            records += profile_rows
            pb.update()

    metrics = pd.DataFrame.from_records(records)
    logger.info("measured recs for %d profiles", metrics["profile_id"].nunique())

    profile_out_fn = project_root() / "outputs" / eval_name / "profile-metrics.csv"
    logger.info("saving per-profile metrics to %s", profile_out_fn)
    metrics.to_csv(profile_out_fn)

    base_metrics = metrics.drop(columns=["profile_id", "personalized"])

    agg_main = (
        base_metrics.drop(columns=["prompt_level"], inplace=False)
        .groupby(["recommender", "theta_topic", "theta_loc", "similarity_threshold"])
        .mean()
    )

    event_filtered = base_metrics[base_metrics["prompt_level"] == "event"]
    agg_event = (
        event_filtered.drop(columns=["prompt_level"], inplace=False)
        .groupby(["recommender", "theta_topic", "theta_loc", "similarity_threshold"])[["rouge1", "rouge2", "rougeL"]]
        .mean()
    )
    agg_event.columns = ["rouge1_event", "rouge2_event", "rougeL_event"]

    topic_filtered = base_metrics[base_metrics["prompt_level"] == "topic"]
    agg_topic = (
        topic_filtered.drop(columns=["prompt_level"], inplace=False)
        .groupby(["recommender", "theta_topic", "theta_loc", "similarity_threshold"])[["rouge1", "rouge2", "rougeL"]]
        .mean()
    )
    agg_topic.columns = ["rouge1_topic", "rouge2_topic", "rougeL_topic"]

    agg_metrics = agg_main.drop(
        columns=["rouge1", "rouge2", "rougeL"],
        errors="ignore",
    ).join([agg_event, agg_topic], how="left")

    # agg_metrics = (
    #     metrics.drop(columns=["profile_id", "personalized"])
    #     .groupby(["recommender", "theta_topic", "theta_loc", "similarity_threshold"])
    #     .mean()
    # )
    # # reciprocal rank means to MRR
    agg_metrics = agg_metrics.rename(columns={"RR": "MRR"})

    logger.info("aggregate metrics:\n%s", agg_metrics)

    out_fn = project_root() / "outputs" / eval_name / "metrics.csv"
    logger.info("saving evaluation to %s", out_fn)
    agg_metrics.to_csv(out_fn)


if __name__ == "__main__":
    main()
