#!/usr/bin/env python3
"""
[INPUT]: 依赖 InterFuser 五个下游输出张量与 CarlaMVDetDataset 的七元 target，消费预注册 occupancy/有效 waypoint 阈值。
[OUTPUT]: 对外提供 MetricError、binary_confusion_metrics、binary_score_metrics 与 InterfuserMetricAccumulator，归约交通栅格、轨迹、路口、红灯和停车标志指标。
[POS]: tools/evaluation 的纯离线指标层；不加载模型或数据集，使冻结 test 的数学口径可在 GPU runner 之外独立回归。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import math

import numpy as np


class MetricError(ValueError):
    """Raised when test outputs cannot support the frozen metric contract."""


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _require_binary(values, label):
    values = _array(values).reshape(-1)
    if values.size == 0:
        raise MetricError(f"{label} is empty")
    if not np.isin(values, (0, 1)).all():
        raise MetricError(f"{label} must contain only binary values")
    return values.astype(np.int64, copy=False)


def binary_confusion_metrics(targets, predictions, class_names):
    """Return deterministic binary confusion, per-class metrics and macro means."""
    targets = _require_binary(targets, "targets")
    predictions = _require_binary(predictions, "predictions")
    if targets.shape != predictions.shape:
        raise MetricError("targets and predictions have different shapes")
    if not isinstance(class_names, (tuple, list)) or len(class_names) != 2:
        raise MetricError("class_names must define exactly two classes")

    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []
    for index, name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = _ratio(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[str(name)] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "samples": int(targets.size),
        "confusion_matrix_target_rows": confusion.tolist(),
        "accuracy": _ratio(int(np.trace(confusion)), int(confusion.sum())),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
    }


def _threshold_curve(targets, scores):
    order = np.argsort(-scores, kind="mergesort")
    targets = targets[order]
    scores = scores[order]
    true_positives = np.cumsum(targets, dtype=np.int64)
    false_positives = np.cumsum(1 - targets, dtype=np.int64)
    distinct_ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    return true_positives[distinct_ends], false_positives[distinct_ends]


def binary_score_metrics(targets, scores, threshold):
    """Return fixed-threshold metrics plus threshold-independent AP and ROC-AUC."""
    targets = _require_binary(targets, "score targets")
    scores = _array(scores).reshape(-1).astype(np.float64, copy=False)
    if targets.shape != scores.shape:
        raise MetricError("score targets and scores have different shapes")
    if not np.isfinite(scores).all():
        raise MetricError("scores contain non-finite values")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise MetricError("threshold must be finite")
    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    if positives == 0 or negatives == 0:
        raise MetricError("binary score metrics require both target classes")

    true_positives, false_positives = _threshold_curve(targets, scores)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positives
    previous_recall = np.r_[0.0, recall[:-1]]
    average_precision = float(np.sum((recall - previous_recall) * precision))
    true_positive_rate = np.r_[0.0, true_positives / positives]
    false_positive_rate = np.r_[0.0, false_positives / negatives]
    roc_auc = float(
        np.sum(
            np.diff(false_positive_rate)
            * (true_positive_rate[1:] + true_positive_rate[:-1])
            * 0.5
        )
    )

    result = binary_confusion_metrics(
        targets,
        (scores >= float(threshold)).astype(np.int64),
        ("empty", "occupied"),
    )
    confusion = result["confusion_matrix_target_rows"]
    intersection = confusion[1][1]
    union = confusion[1][1] + confusion[0][1] + confusion[1][0]
    result.update(
        {
            "threshold": float(threshold),
            "average_precision": average_precision,
            "roc_auc": roc_auc,
            "occupied_iou": _ratio(intersection, union),
        }
    )
    return result


class InterfuserMetricAccumulator:
    """Accumulate exact dataset-level metrics without averaging batch averages."""

    ATTRIBUTE_NAMES = ("offset_x", "offset_y", "yaw", "extent_x", "extent_y")
    CLASS_HEADS = {
        "junction": (2, 2, ("not_junction", "junction")),
        "red_light": (3, 3, ("red", "not_red")),
        "stop_sign": (4, 6, ("absent", "present")),
    }

    def __init__(
        self,
        traffic_positive_target_threshold=0.01,
        traffic_prediction_threshold=0.5,
        invalid_waypoint_threshold=1000.0,
    ):
        self.positive_threshold = float(traffic_positive_target_threshold)
        self.prediction_threshold = float(traffic_prediction_threshold)
        self.invalid_waypoint_threshold = float(invalid_waypoint_threshold)
        for value, label in (
            (self.positive_threshold, "traffic positive threshold"),
            (self.prediction_threshold, "traffic prediction threshold"),
            (self.invalid_waypoint_threshold, "invalid waypoint threshold"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise MetricError(f"{label} must be finite and positive")

        self.samples = 0
        self._traffic_targets = []
        self._traffic_scores = []
        self._traffic_probability_abs_error = 0.0
        self._traffic_positive_probability_abs_error = 0.0
        self._traffic_negative_prediction_sum = 0.0
        self._traffic_positive_cells = 0
        self._traffic_negative_cells = 0
        self._attribute_abs_error = np.zeros(5, dtype=np.float64)
        self._speed_abs_error = 0.0
        self._waypoint_distance_sum = np.zeros(10, dtype=np.float64)
        self._waypoint_coordinate_abs_sum = np.zeros(10, dtype=np.float64)
        self._waypoint_support = np.zeros(10, dtype=np.int64)
        self._class_targets = {name: [] for name in self.CLASS_HEADS}
        self._class_predictions = {name: [] for name in self.CLASS_HEADS}

    def update(self, outputs, targets):
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 5:
            raise MetricError("InterFuser outputs must contain at least five heads")
        if not isinstance(targets, (tuple, list)) or len(targets) < 7:
            raise MetricError("InterFuser targets must contain seven entries")

        traffic_output = _array(outputs[0]).astype(np.float64, copy=False)
        traffic_target = _array(targets[4]).astype(np.float64, copy=False)
        if traffic_output.shape != traffic_target.shape or traffic_output.ndim != 3:
            raise MetricError("traffic output/target shapes differ or are not Bx400x7")
        if traffic_output.shape[1:] != (400, 7):
            raise MetricError(f"unexpected traffic shape: {traffic_output.shape}")
        if not np.isfinite(traffic_output).all() or not np.isfinite(traffic_target).all():
            raise MetricError("traffic tensors contain non-finite values")
        batch_size = traffic_output.shape[0]
        self.samples += batch_size

        target_probability = traffic_target[:, :, 0]
        predicted_probability = traffic_output[:, :, 0]
        positive = target_probability >= self.positive_threshold
        negative = ~positive
        self._traffic_targets.append(positive.reshape(-1).astype(np.int8))
        self._traffic_scores.append(predicted_probability.reshape(-1).astype(np.float32))
        self._traffic_probability_abs_error += float(
            np.abs(predicted_probability - target_probability).sum()
        )
        self._traffic_positive_cells += int(positive.sum())
        self._traffic_negative_cells += int(negative.sum())
        self._traffic_positive_probability_abs_error += float(
            np.abs(predicted_probability[positive] - target_probability[positive]).sum()
        )
        self._traffic_negative_prediction_sum += float(predicted_probability[negative].sum())
        if positive.any():
            attribute_error = np.abs(
                traffic_output[:, :, 1:6][positive] - traffic_target[:, :, 1:6][positive]
            )
            self._attribute_abs_error += attribute_error.sum(axis=0)
            self._speed_abs_error += float(
                np.abs(traffic_output[:, :, 6][positive] - traffic_target[:, :, 6][positive]).sum()
            )

        waypoint_output = _array(outputs[1]).astype(np.float64, copy=False)
        waypoint_target = _array(targets[1]).astype(np.float64, copy=False)
        if waypoint_output.shape != waypoint_target.shape or waypoint_output.shape[1:] != (10, 2):
            raise MetricError("waypoint output/target shapes differ or are not Bx10x2")
        if not np.isfinite(waypoint_output).all():
            raise MetricError("waypoint outputs contain non-finite values")
        valid_waypoints = np.isfinite(waypoint_target).all(axis=2) & (
            np.abs(waypoint_target) < self.invalid_waypoint_threshold
        ).all(axis=2)
        waypoint_abs_error = np.abs(waypoint_output - waypoint_target)
        waypoint_distance = np.linalg.norm(waypoint_output - waypoint_target, axis=2)
        for horizon in range(10):
            valid = valid_waypoints[:, horizon]
            self._waypoint_support[horizon] += int(valid.sum())
            self._waypoint_distance_sum[horizon] += float(
                waypoint_distance[valid, horizon].sum()
            )
            self._waypoint_coordinate_abs_sum[horizon] += float(
                waypoint_abs_error[valid, horizon].sum()
            )

        for name, (output_index, target_index, _) in self.CLASS_HEADS.items():
            logits = _array(outputs[output_index])
            labels = _array(targets[target_index]).reshape(-1)
            if logits.shape != (batch_size, 2) or labels.shape != (batch_size,):
                raise MetricError(f"{name} head has an unexpected shape")
            if not np.isfinite(logits).all():
                raise MetricError(f"{name} logits contain non-finite values")
            self._class_targets[name].append(_require_binary(labels, f"{name} targets"))
            self._class_predictions[name].append(np.argmax(logits, axis=1).astype(np.int64))

    def finalize(self):
        if self.samples <= 0:
            raise MetricError("no samples were accumulated")
        traffic_targets = np.concatenate(self._traffic_targets)
        traffic_scores = np.concatenate(self._traffic_scores)
        grid_cells = self._traffic_positive_cells + self._traffic_negative_cells
        if grid_cells != self.samples * 400:
            raise MetricError("traffic grid cell count is inconsistent")
        if self._traffic_positive_cells == 0 or self._traffic_negative_cells == 0:
            raise MetricError("traffic occupancy requires positive and negative cells")
        if np.any(self._waypoint_support == 0):
            raise MetricError("every waypoint horizon requires test support")

        distance_per_horizon = self._waypoint_distance_sum / self._waypoint_support
        coordinate_mae_per_horizon = self._waypoint_coordinate_abs_sum / (
            self._waypoint_support * 2
        )
        waypoint_total_support = int(self._waypoint_support.sum())
        metrics = {
            "samples": self.samples,
            "traffic": {
                "grid_cells": grid_cells,
                "positive_cells": self._traffic_positive_cells,
                "negative_cells": self._traffic_negative_cells,
                "occupancy": binary_score_metrics(
                    traffic_targets, traffic_scores, self.prediction_threshold
                ),
                "probability_mae": self._traffic_probability_abs_error / grid_cells,
                "positive_probability_mae": self._traffic_positive_probability_abs_error
                / self._traffic_positive_cells,
                "negative_prediction_mean": self._traffic_negative_prediction_sum
                / self._traffic_negative_cells,
                "attribute_mae_positive_cells": {
                    name: float(self._attribute_abs_error[index] / self._traffic_positive_cells)
                    for index, name in enumerate(self.ATTRIBUTE_NAMES)
                },
                "speed_mae_positive_cells": self._speed_abs_error
                / self._traffic_positive_cells,
            },
            "waypoints": {
                "support_per_horizon": self._waypoint_support.tolist(),
                "mean_distance_per_horizon": distance_per_horizon.tolist(),
                "coordinate_mae_per_horizon": coordinate_mae_per_horizon.tolist(),
                "ade": float(self._waypoint_distance_sum.sum() / waypoint_total_support),
                "coordinate_mae": float(
                    self._waypoint_coordinate_abs_sum.sum() / (waypoint_total_support * 2)
                ),
                "fde_horizon_10": float(distance_per_horizon[-1]),
            },
        }
        for name, (_, _, class_names) in self.CLASS_HEADS.items():
            targets = np.concatenate(self._class_targets[name])
            if not np.all(np.bincount(targets, minlength=2) > 0):
                raise MetricError(f"{name} requires both classes in frozen test")
            metrics[name] = binary_confusion_metrics(
                targets, np.concatenate(self._class_predictions[name]), class_names
            )
        return metrics
