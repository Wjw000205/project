from __future__ import annotations

import subprocess

from src import web_visualizer


class FakeProcess:
    pid = 12345

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> None:
        return None

    def kill(self) -> None:
        self.killed = True


def test_stop_training_marks_job_and_terminates_process_tree(monkeypatch, tmp_path) -> None:
    fake = FakeProcess()
    log_path = tmp_path / "train.log"
    log_path.write_text("", encoding="utf-8")
    web_visualizer.JOBS["demo-job"] = {
        "job_id": "demo-job",
        "process": fake,
        "cmd": ["python"],
        "config_path": "config.yaml",
        "run_dir": "outputs/demo-job",
        "log_path": log_path.as_posix(),
        "started_at": 1.0,
    }
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(web_visualizer.subprocess, "run", fake_run)

    result = web_visualizer._stop_training("demo-job")

    assert result["state"] == "stopping"
    assert web_visualizer.JOBS["demo-job"]["stop_requested"] is True
    assert "Stop requested" in log_path.read_text(encoding="utf-8")
    if web_visualizer.os.name == "nt":
        assert calls == [["taskkill", "/PID", "12345", "/T", "/F"]]
    else:
        assert fake.terminated is True


def test_stop_training_reports_finished_job_without_terminating(tmp_path) -> None:
    class FinishedProcess(FakeProcess):
        def poll(self):
            return 0

    fake = FinishedProcess()
    log_path = tmp_path / "train.log"
    log_path.write_text("", encoding="utf-8")
    web_visualizer.JOBS["finished-job"] = {
        "job_id": "finished-job",
        "process": fake,
        "cmd": ["python"],
        "config_path": "config.yaml",
        "run_dir": "outputs/finished-job",
        "log_path": log_path.as_posix(),
        "started_at": 1.0,
    }

    result = web_visualizer._stop_training("finished-job")

    assert result["state"] == "finished"
    assert fake.terminated is False


def test_active_training_jobs_adopts_running_web_run(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "web_visualizer" / "runs" / "active-job"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("exp:\n  name: active\n", encoding="utf-8")
    (run_dir / "train.log").write_text(
        "Train ETTm1 H=96 [#---------------------------] 10/100  10.0% "
        "elapsed=00:01 eta=00:09 | epoch=1/10 batch=10/10 loss=0.1",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_visualizer, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(web_visualizer, "_find_running_train_pids", lambda path: [12345])
    monkeypatch.setattr(web_visualizer, "_pid_running", lambda pid: True)
    web_visualizer.JOBS.pop("active-job", None)

    jobs = web_visualizer._active_training_jobs()

    assert [job["job_id"] for job in jobs] == ["active-job"]
    assert jobs[0]["state"] == "running"
    assert jobs[0]["progress"]["epoch_current"] == 1


def test_finished_job_status_marks_progress_complete(tmp_path) -> None:
    class FinishedProcess(FakeProcess):
        def poll(self):
            return 0

    log_path = tmp_path / "train.log"
    log_path.write_text(
        "Train ETTm1 H=96 [##########------------------] 18156/53400  34.0% "
        "elapsed=13:36 eta=26:24 | epoch=34/100 loss=0.391\n"
        "Saved run summary to: outputs/demo/run_summary.json\n",
        encoding="utf-8",
    )
    web_visualizer.JOBS["terminal-job"] = {
        "job_id": "terminal-job",
        "process": FinishedProcess(),
        "cmd": ["python"],
        "config_path": "config.yaml",
        "run_dir": "outputs/terminal-job",
        "log_path": log_path.as_posix(),
        "started_at": 1.0,
    }

    status = web_visualizer._job_status("terminal-job")

    assert status["state"] == "finished"
    assert status["progress"]["phase"] == "finished"
    assert status["progress"]["global_percent"] == 100.0
    assert status["progress"]["epoch_percent"] == 100.0
    assert status["progress"]["display_percent"] == 100.0
    assert status["progress"]["raw"] == web_visualizer.TRAINING_FINISHED_TEXT


def test_running_status_has_log_placeholder_when_tail_only_contains_progress(tmp_path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "Train ETTm1 H=96 [#---------------------------] 10/100  10.0% "
        "elapsed=00:01 eta=00:09 | epoch=1/10 batch=10/10 loss=0.1",
        encoding="utf-8",
    )
    web_visualizer.JOBS["progress-only-job"] = {
        "job_id": "progress-only-job",
        "process": FakeProcess(),
        "cmd": ["python"],
        "config_path": "config.yaml",
        "run_dir": "outputs/progress-only-job",
        "log_path": log_path.as_posix(),
        "started_at": 1.0,
    }

    status = web_visualizer._job_status("progress-only-job")

    assert status["state"] == "running"
    assert status["log_tail"] == web_visualizer.LOG_PROGRESS_PLACEHOLDER


def test_validating_progress_display_uses_global_percent_not_epoch_percent(tmp_path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "Train ETTm1 H=96 [####------------------------] 1068/53400  2.0% "
        "elapsed=00:40 eta=32:20 | epoch=2/100 loss=0.42 validating",
        encoding="utf-8",
    )
    web_visualizer.JOBS["validating-job"] = {
        "job_id": "validating-job",
        "process": FakeProcess(),
        "cmd": ["python"],
        "config_path": "config.yaml",
        "run_dir": "outputs/validating-job",
        "log_path": log_path.as_posix(),
        "started_at": 1.0,
    }

    status = web_visualizer._job_status("validating-job")

    assert status["state"] == "running"
    assert status["progress"]["phase"] == "validating"
    assert status["progress"]["epoch_percent"] == 100.0
    assert status["progress"]["global_percent"] == 2.0
    assert status["progress"]["display_percent"] == 2.0


def test_index_formatter_keeps_text_metrics_visible() -> None:
    html = web_visualizer.INDEX_HTML

    assert 'if (typeof value === "string") return value || "-";' in html
    assert 'if (Array.isArray(value)) return value.length ? value.join(", ") : "-";' in html
    assert 'Number.isNaN(Number(value))' not in html


def test_prediction_sample_controls_include_carousel_arrows() -> None:
    html = web_visualizer.INDEX_HTML

    assert 'id="prevSampleBtn"' in html
    assert 'id="nextSampleBtn"' in html
    assert "changeSample(-1)" in html
    assert "changeSample(1)" in html


def test_prediction_sample_controls_use_aligned_toolbar_rows() -> None:
    html = web_visualizer.INDEX_HTML

    assert "prediction-controls" in html
    assert "grid-template-rows: 20px 34px 18px" in html
    assert "control-label-spacer" in html
    assert 'style="display:flex;align-items:end"' not in html


def test_training_panel_can_be_hidden_and_shown_again() -> None:
    html = web_visualizer.INDEX_HTML

    assert 'id="showJobPanelBtn"' in html
    assert 'showTrainingPanel()' in html
    assert 'hideTrainingPanel()' in html
