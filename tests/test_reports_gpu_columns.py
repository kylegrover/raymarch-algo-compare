import os
from raymarching_benchmark.metrics.analyzer import MetricsAnalyzer
from raymarching_benchmark.core.types import RayMarchStats
from raymarching_benchmark.visualization.reports import generate_markdown_report


def test_report_includes_gpu_columns_when_stats_have_gpu(tmp_path):
    out_dir = tmp_path / "results"
    out_dir.mkdir()

    analyzer = MetricsAnalyzer()
    # create a minimal RayMarchStats with GPU fields populated
    stat = RayMarchStats(strategy_name="Segment", scene_name="FakeScene")
    stat.iteration_mean = 3.14
    stat.hit_rate = 0.5
    stat.iteration_p95 = 5.0
    stat.warp_divergence_proxy = 0.1
    stat.time_per_ray_us = 12.0
    stat.gpu_time_per_ray_us = 0.123
    stat.gpu_warp_divergence_proxy = 0.45

    analyzer.add_result(stat)

    report_path = generate_markdown_report(analyzer, str(out_dir))
    assert os.path.exists(report_path)

    txt = open(report_path, 'r', encoding='utf-8').read()
    # Aggregated table must include GPU Time column
    assert "GPU Time (us/ray)" in txt
    # Per-scene table header should include GPU columns when per-run GPU data exists
    assert "GPU WD" in txt
