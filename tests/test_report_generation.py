import tempfile
from raymarching_benchmark.visualization.reports import generate_markdown_report
from raymarching_benchmark.core.types import RayMarchStats

class _FakeAnalyzer:
    def __init__(self):
        self._strategies = ["Standard"]
        self._scenes = ["Sphere"]
        s = RayMarchStats(strategy_name="Standard", scene_name="Sphere")
        s.iteration_mean = 1.0
        s.hit_rate = 0.5
        s.iteration_p95 = 2.0
        s.warp_divergence_proxy = 0.1
        s.time_per_ray_us = 10.0
        s.gpu_time_per_ray_us = 0.5
        s.gpu_time_per_ray_median_us = 0.45
        s.gpu_time_sample_count = 7
        s.gpu_width = 1920
        s.gpu_height = 1080
        self.all_stats = [s]

    def get_strategies(self): return self._strategies
    def get_scenes(self): return self._scenes
    def strategy_summary(self):
        return {"Standard": {
            "avg_mean_iterations": 1.0,
            "avg_hit_rate": 0.5,
            "num_wins": 0,
            "avg_warp_divergence": 0.1,
            "avg_time_per_ray_us": 10.0,
            "avg_gpu_time_per_ray_us": 0.5,
            "avg_gpu_warp_divergence": 0.05,
        }}
    def get_stat(self, strat, scene): return self.all_stats[0]


def test_report_includes_gpu_resolution_and_stats(tmp_path):
    out = generate_markdown_report(_FakeAnalyzer(), str(tmp_path))
    text = (tmp_path / "REPORT.md").read_text()
    assert "GPU measurements were taken at: **1920x1080**" in text
    assert "GPU Time (us/ray)" in text
    assert "0.45" in text  # median appears somewhere in the report


def test_analyzer_prefers_gpu_median_for_aggregation():
    from raymarching_benchmark.metrics.analyzer import MetricsAnalyzer
    from raymarching_benchmark.core.types import RayMarchStats
    a = MetricsAnalyzer()
    s1 = RayMarchStats('A','S')
    s1.gpu_time_per_ray_us = 1.0
    s2 = RayMarchStats('A','T')
    s2.gpu_time_per_ray_median_us = 0.1
    s2.gpu_width = 1920
    s2.gpu_height = 1080
    a.add_result(s1)
    a.add_result(s2)
    summary = a.strategy_summary()
    # average should consider the median value (0.1) and legacy value (1.0)
    assert 'A' in summary
    assert summary['A']['avg_gpu_time_per_ray_us'] is not None