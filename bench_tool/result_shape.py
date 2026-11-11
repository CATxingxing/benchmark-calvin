"""benchmark.controller/run 的 ToolResult 输出契约（七个必填字段 + 可选扩展）。

字段集合与 skill 树 gateway.yaml 里的 output_schema 一一对应：
多一个少一个都会让下游 schema 校验失败，改这里必须同时改 skill 树。

outputs.status 是**给 Agent 看的批次结论**，枚举 {succeeded, failed, partial,
stopped}；ToolResult.status 是 forge 的执行结论，只能是 {succeeded, failed,
stopped}。批次被 stop 时两者分叉：completed_episodes>0 → partial，
一集没跑完 → stopped。

可选扩展（调用方传了才出现，七字段的老 schema 不受影响）：suite、
primary_metric、metrics、is_full_suite、args。CALVIN 的 avg_seq_len / SR1–SR5
就在 metrics 里，Agent 不用再去读 result_file。
"""

from __future__ import annotations

from typing import Any

from _core.result import BenchmarkResult


def stopped_summary(*, completed: int, total: int, reason: str | None) -> str:
    summary = f"stopped after {completed}/{total} episodes"
    return f"{summary}（{reason}）" if reason else summary


def build_outputs(
    *,
    run_id: str,
    batch_status: str,
    completed_episodes: int,
    total_episodes: int,
    success_rate: float,
    result_file: str | None,
    failure_summary: str | None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = batch_status
    if batch_status == "stopped":
        status = "partial" if completed_episodes > 0 else "stopped"
    outputs = {
        "run_id": run_id,
        "status": status,
        "completed_episodes": completed_episodes,
        "total_episodes": total_episodes,
        "success_rate": success_rate,
        "result_file": result_file,
        "failure_summary": failure_summary,
    }
    for name, value in (extras or {}).items():
        if name in outputs:
            raise ValueError(f"extras 不能覆盖必填字段 {name!r}")
        outputs[name] = value
    return outputs


def is_full_suite(*, plan: dict[str, Any], capabilities: dict[str, Any], completed_episodes: int, status: str) -> bool:
    """这次 run 是否覆盖了所选 suite 的全部 task × init state，且每集都跑完。

    只看覆盖面，不判断数据集本身是否官方全量（例如 calvin_debug 数据集或
    smoke 配置下 suite 本来就更小）；那部分由 skill 的 profile 保证。
    """
    if status != "succeeded":
        return False
    suite = plan.get("suite")
    tasks = capabilities.get("tasks") or []
    if suite is not None and suite != capabilities.get("suite"):
        tasks = next(
            (item["tasks"] for item in capabilities.get("suites") or [] if item.get("suite") == suite),
            [],
        )
    if not tasks:
        return False
    counts = {int(t["task_id"]): int(t["num_init_states"]) for t in tasks}
    if set(int(t) for t in plan.get("task_ids") or []) != set(counts):
        return False
    init_ids = set(int(i) for i in plan.get("init_state_ids") or [])
    if any(not set(range(n)) <= init_ids for n in counts.values()):
        return False
    return completed_episodes == int(plan.get("total_episodes") or -1) and completed_episodes > 0


def outputs_from_result(
    *,
    run_id: str,
    result: BenchmarkResult,
    result_file: str | None,
    failure_summary: str | None,
    plan: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = result.to_dict()
    extras: dict[str, Any] | None = None
    if plan is not None and capabilities is not None:
        completed = int(payload.get("completed_episodes") or 0)
        extras = {
            "suite": result.suite or plan.get("suite"),
            "primary_metric": dict(result.primary_metric or {}),
            "metrics": dict(result.metrics or {}),
            "is_full_suite": is_full_suite(
                plan=plan, capabilities=capabilities, completed_episodes=completed, status=result.status
            ),
            "args": {k: plan[k] for k in ("suite", "task_ids", "init_state_ids", "num_runs", "seed", "max_steps") if k in plan},
        }
    return build_outputs(
        run_id=run_id,
        batch_status=result.status,
        completed_episodes=int(payload.get("completed_episodes") or 0),
        total_episodes=int(payload.get("total_episodes") or 0),
        success_rate=float(payload.get("success_rate") or 0.0),
        result_file=result_file,
        failure_summary=failure_summary,
        extras=extras,
    )


__all__ = ["build_outputs", "is_full_suite", "outputs_from_result", "stopped_summary"]
