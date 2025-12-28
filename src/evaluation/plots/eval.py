from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classes.classes import ModelConfig
from classes.paths import LocalPaths
from classes.workloads import EvaluationWorkload, JoinOrderEvalWorkload, LiteralEvalWorkload, \
    JoinImplEvalWorkload
from training.dataset.dataset_creation import read_workload_runs
from evaluation.plots.evaluation_metrics import Metric, PickRate


class Evaluator:
    def __init__(self):
        self.metric_collection = []
        self.minimal_runtimes = dict()

    @staticmethod
    def plot_single_experiment(models: List[ModelConfig],
                               results: pd.DataFrame,
                               metric_results: List[dict],
                               metrics: List[Metric],
                               workload: EvaluationWorkload,
                               plot_cardinalities: bool = False,
                               plot_metrics: bool = False):
        fontsize = 12
        # _, (ax, ax1) = plt.subplots(1,2)
        fig = plt.figure(figsize=(10, 5), dpi=100)
        if plot_metrics:
            ax = plt.subplot2grid((1, 3), (0, 0), rowspan=2, colspan=2, fig=fig)
            ax_small1 = plt.subplot2grid((3, 3), (0, 2))
            ax_small2 = plt.subplot2grid((3, 3), (1, 2), sharex=ax_small1)
            ax_small3 = plt.subplot2grid((3, 3), (2, 2), sharex=ax_small1)

        if isinstance(workload, LiteralEvalWorkload):
            # if not workload.is_categorical():
            results["query_index"] = workload.get_y_range()
            ax.set_xlabel(workload.column)

        elif isinstance(workload, JoinImplEvalWorkload):
            ax.set_xlim(-0.1, 2.1)
            ax.xaxis.set_ticks(np.arange(0, 3, 1))
            ax.set_xticklabels(workload.yticks, fontsize=fontsize)

        if plot_metrics:
            for metric, sub_ax in zip(metrics, [ax_small1, ax_small2, ax_small3]):
                metric_dc = pd.DataFrame.from_dict(metric_results)[["model_name", metric.metric_name]].set_index(
                    "model_name")
                if metric.metric_name == PickRate().metric_name:
                    metric_dc[metric.metric_name] = metric_dc[metric.metric_name].astype(int)
                sub_ax_t = sub_ax.twinx()

                metric_dc[metric.metric_name].plot(kind='bar',
                                                   ax=sub_ax_t,
                                                   color=[m.color for m in models],
                                                   zorder=3,
                                                   width=0.8,
                                                   edgecolor=None)
                sub_ax_t.grid(zorder=0)
                sub_ax_t.set_ylabel(metric.display_name, fontsize=fontsize)
                sub_ax_t.tick_params(axis='x', which='both', labelsize=fontsize)
                sub_ax_t.tick_params(axis='y', which='both', labelsize=fontsize)
                sub_ax_t.set_ylim(metric.y_min, metric.y_max)
                sub_ax.set_xticklabels([])
                # sub_ax_t.bar_label(sub_ax_t.containers[0], label_type='edge', fmt='{:,.2f}', rotation=90, padding=2, fontsize=9)
                sub_ax.yaxis.set_visible(False)
                if metric.logscale:
                    sub_ax_t.set_yscale("log")

        for model, metric in zip(models, metric_results):
            label = model.display_name
            model_results = results[results["model"] == model.display_name]

            # Sort results by runtime for join order workloads
            if isinstance(workload, JoinOrderEvalWorkload):
                model_results = model_results.sort_values(by="runtime").reset_index(drop=False)
                model_results.reset_index(drop=True, inplace=True)
                model_results['query_index'] = model_results.index

            model_results.plot(x="query_index",
                               y="prediction",
                               ax=ax,
                               linestyle="-",
                               label=label,
                               color=model.color,
                               linewidth=3)

            model_results.plot.scatter(x="query_index",
                                       y="prediction",
                                       ax=ax,
                                       color=model.color)

        # Plot real runtime
        model_results.plot(x="query_index", y="runtime", linestyle="-", ax=ax, label="Real Runtime", color="black",
                           linewidth=3)
        model_results.plot.scatter(x="query_index", y="runtime", ax=ax, s=20, color="black")

        if isinstance(workload, JoinOrderEvalWorkload):
            ax.set_xlabel('Join Enumeration')
            ax.set_xticks(ax.get_xticks(), fontsize=fontsize)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=fontsize, ha='right')

        # Plot cardinalities
        if plot_cardinalities:
            ax2 = ax.twinx()
            ax2.tick_params(axis='y', labelcolor="black")
            model_results.plot(x="query_index", y="act_card", linestyle="-", ax=ax2, label="Σ Act. Cardinality",
                               color="gray",
                               linewidth=3)
            model_results.plot.scatter(x="query_index", y="act_card", ax=ax2, color="gray")

            model_results.plot(x="query_index", y="est_card", linestyle="-", ax=ax2, label="Σ Est. Cardinality",
                               color="lightgray", linewidth=3)
            model_results.plot.scatter(x="query_index", y="est_card", ax=ax2, color="lightgray", marker='x')

            ax2.set_ylabel('Cardinality')

            # Merge legends to common legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, ncol=2, loc="upper left",
                       prop={'family': 'monospace', 'size': 6})
            ax.get_legend().remove()

        if isinstance(workload, LiteralEvalWorkload):
            ax.set_xlabel(workload.column, fontsize=fontsize)
        elif isinstance(workload, JoinOrderEvalWorkload):
            ax.set_xlabel('Join Enumeration', fontsize=fontsize)
        elif isinstance(workload, JoinImplEvalWorkload):
            ax.set_xlabel('Operator Type', fontsize=fontsize)
        ax.set_yscale('log')
        ax.set_ylabel("Runtime (s)")
        ax.legend(ncol=2, fontsize=fontsize)
        ax.set_xlim([-0.1, len(model_results) - 1 + 0.1])
        ax.grid(which='minor', axis='y')

        ax.set_title(workload.get_query_label(), fontdict={'family': 'monospace', 'size': fontsize})
        plt.tight_layout()
        fig.align_labels()
        plt.subplots_adjust(wspace=0.25, hspace=0.13)
        # plt.show()
        path = Path(f"{LocalPaths().data}/plots/{workload.folder}/{workload.database.db_name}")
        if not path.exists():
            path.mkdir(parents=True)
        # plt.savefig(f"{path/workload.get_workload_name()}.pdf")
        plt.show()
        plt.close()

    def eval(self,
             workloads: List[EvaluationWorkload],
             metrics: List[Metric],
             plot_single_workloads: bool = True,
             plot_limit: int = np.inf,
             model_configs: List[ModelConfig] = [],
             seeds: List[int] = []):

        wl_summaries = []
        result_collection = []
        plot_count = 0
        for workload in workloads:
            # Readout workload
            plans, database_statistics = read_workload_runs(
                workload_run_paths=[workload.get_workload_path(LocalPaths().parsed_plans)])

            # Get cardinalities
            cardinalities = Evaluator.readout_workload_information(plans, workload)

            # Combine model predictions to common dataframe
            predictions = Evaluator.combine_predictions(model_configs, workload, seeds)

            if workload.database.db_name not in self.minimal_runtimes:
                self.minimal_runtimes[workload.database.db_name] = []
            self.minimal_runtimes[workload.database.db_name].append(predictions["label"].min())

            # Aggregate and reduce the predictions over the given seeds
            predictions = predictions.groupby(['model', 'query_index', 'label'])['prediction'].mean().reset_index()

            # Merge cardinalities with predictions
            results = pd.merge(predictions, cardinalities, on=["query_index"])

            # Sort results by runtime for join order workloads
            if isinstance(workload, JoinOrderEvalWorkload):
                results = results.sort_values(by="runtime").reset_index(drop=False)
                results.reset_index(drop=True, inplace=True)
                results['query_index'] = results.index

            # Collect results for later reusage
            result_collection.append((workload, results))

            # Compute evaluation metrics
            all_metrics = []
            for model in model_configs:
                preds, labels = predictions[predictions["model"] == model.name.DISPLAY_NAME]["prediction"], \
                predictions[predictions["model"] == model.name.DISPLAY_NAME]["label"]
                if not preds.empty and not labels.empty:
                    metric = dict(model_name=model.name.DISPLAY_NAME,
                                  workload=workload.get_workload_name(),
                                  workload_type=workload.folder,
                                  database=workload.database.db_name)

                    if isinstance(workload, JoinOrderEvalWorkload):
                        #    metric["table_size"] = workload.get_table_length()
                        #    metric["unique_values"] = workload.get_column_unique()
                        metric["num_tables"] = workload.num_tables

                    for m in metrics:
                        metric[m.metric_name] = m.evaluate_metric(preds=preds, labels=labels)
                    all_metrics.append(metric)

            # Plot single workload plots
            if plot_single_workloads and plot_count < plot_limit:
                plot_count += 1
                Evaluator.plot_single_experiment(models=model_configs,
                                                 results=results,
                                                 metric_results=all_metrics,
                                                 workload=workload,
                                                 metrics=metrics,
                                                 plot_metrics=True)
            self.metric_collection += all_metrics

    @staticmethod
    def combine_predictions(models: list[ModelConfig], workload: EvaluationWorkload, seeds: List[int], retrain=False) -> pd.DataFrame:
        # Reading out and merge different model predictions
        results = []
        for seed in seeds:
            for model in models:
                prediction_path = Path(f"{model.get_eval_dir(source_path=LocalPaths(), database=workload.database, retrain=retrain)}/"
                                       f"{workload.folder}/"
                                       f"{workload.get_workload_name()}_{seed}_test_pred.csv")
                assert prediction_path.exists(), f"Prediction file {prediction_path} does not exist"
                res = pd.read_csv(prediction_path).sort_values(by=['query_index'])
                res["model"] = model.name.DISPLAY_NAME
                res = res.drop(columns=["qerror"])
                res['seed'] = seed
                results.append(res)
        df = pd.concat(results, ignore_index=True)
        return df

    @staticmethod
    def find_join_type_recursively(plans):
        for plan in plans:
            if "Join" in plan.plan_parameters.op_name or "NestLoop" in plan.plan_parameters.op_name:
                return plan.plan_parameters.op_name
            else:
                return Evaluator.find_join_type_recursively(plan.children)

    @staticmethod
    def get_cardinality_recursively(p, counter=0, missing_card=False, type: str = "act_card"):
        if not hasattr(p.plan_parameters, type):
            missing_card = True
        else:
            card = getattr(p.plan_parameters, type)
            counter += (card * max(1, p.plan_parameters.workers_planned))
        for c in p.children:
            missing_card, children_card = Evaluator.get_cardinality_recursively(c, counter, missing_card)
            counter += children_card
        return missing_card, counter

    @staticmethod
    def readout_workload_information(plans: list, workload: EvaluationWorkload) -> pd.DataFrame:
        # Reading out workload information like runtimes and cardinalities
        missing_card = False
        entries = []
        for idx, plan in enumerate(plans):
            new_missing_card, act_card = Evaluator.get_cardinality_recursively(plan, type="act_card")
            missing_card = new_missing_card or missing_card

            _, est_card = Evaluator.get_cardinality_recursively(plan, type="est_card")

            entries.append(
                dict(runtime=plan.plan_runtime / 1000, act_card=act_card, est_card=est_card, query_index=idx))
        return pd.DataFrame.from_dict(entries)
