"""Tests for convergence monitoring and early stopping."""

import pytest
import numpy as np

from quinkgl.training.convergence import (
    ConvergenceMonitor,
    ConvergenceConfig,
    ConvergenceStatus,
    ConvergenceReport,
)


class TestConvergenceMonitor:
    def test_improving_status_initially(self):
        monitor = ConvergenceMonitor()
        report = monitor.update(10.0, 0.5)
        assert report.status == ConvergenceStatus.IMPROVING

    def test_converged_on_target_accuracy(self):
        config = ConvergenceConfig(target_accuracy=0.95)
        monitor = ConvergenceMonitor(config=config)
        report = monitor.update(0.5, 0.96)
        assert report.status == ConvergenceStatus.CONVERGED
        assert monitor.should_stop(report) is True

    def test_converged_on_target_loss(self):
        config = ConvergenceConfig(target_loss=0.01)
        monitor = ConvergenceMonitor(config=config)
        report = monitor.update(0.005, 0.8)
        assert report.status == ConvergenceStatus.CONVERGED

    def test_plateau_detection(self):
        config = ConvergenceConfig(patience=3, min_delta=0.01, window_size=5)
        monitor = ConvergenceMonitor(config=config)
        for _ in range(10):
            monitor.update(5.0, 0.5)
        report = monitor.update(5.0, 0.5)
        assert report.status in (ConvergenceStatus.PLATEAU, ConvergenceStatus.CONVERGED)

    def test_early_stopping_triggers_after_patience(self):
        config = ConvergenceConfig(patience=2, min_delta=0.01, window_size=3)
        monitor = ConvergenceMonitor(config=config)
        monitor.update(5.0, 0.5)
        monitor.update(5.0, 0.5)
        report = monitor.update(5.0, 0.5)
        assert monitor.should_stop(report) is True

    def test_improvement_resets_patience(self):
        config = ConvergenceConfig(patience=3, min_delta=0.01)
        monitor = ConvergenceMonitor(config=config)
        monitor.update(10.0, 0.1)
        monitor.update(9.5, 0.2)
        report = monitor.update(9.5, 0.2)
        assert report.rounds_without_improvement == 1
        report2 = monitor.update(9.0, 0.3)
        assert report2.rounds_without_improvement == 0

    def test_diverging_detection(self):
        config = ConvergenceConfig(diverging_threshold=0.5, window_size=3)
        monitor = ConvergenceMonitor(config=config)
        for _ in range(5):
            monitor.update(1.0, 0.5)
        report = monitor.update(10.0, 0.1)
        assert report.status == ConvergenceStatus.DIVERGING

    def test_trend_computation(self):
        monitor = ConvergenceMonitor()
        for v in [10.0, 9.0, 8.0, 7.0, 6.0]:
            monitor.update(v, 0.5)
        report = monitor.update(5.0, 0.5)
        assert report.loss_trend == "decreasing"

    def test_best_loss_tracked(self):
        monitor = ConvergenceMonitor()
        monitor.update(10.0, 0.1)
        monitor.update(5.0, 0.2)
        monitor.update(7.0, 0.1)
        assert monitor.best_loss == 5.0

    def test_best_accuracy_tracked(self):
        monitor = ConvergenceMonitor()
        monitor.update(10.0, 0.1)
        monitor.update(5.0, 0.9)
        monitor.update(7.0, 0.5)
        assert monitor.best_accuracy == 0.9
