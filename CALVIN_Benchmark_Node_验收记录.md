## CALVIN ✅ 通过（判据 A）

### 基本信息

| 项 | 内容 |
| :--- | :--- |
| Benchmark | CALVIN |
| 官方仓库 | https://github.com/mees/calvin |
| 官方 Leaderboard | http://calvin.cs.uni-freiburg.de/ |
| 任务/协议 | LH-MTLC（34 子任务，5 子任务链；官方 1000 条序列） |
| 权重 | 官方 MCIL `D_D_static_rgb_baseline`（`mcil_baseline.ckpt`） |
| 模型权重 | http://calvin.cs.uni-freiburg.de/model_weights/D_D_static_rgb_baseline.zip |
| Robot / Action Type | Franka Panda / `absolute_euler`（7D：位置 3 + 欧拉 3 + gripper 1） |
| 集数（每边） | **20** |
| 场景是否同一批 | **是**。Official 与 Forge 均执行 `get_sequences(1000)[:20]`，`seed=0`，逐序列对齐校验通过 |
| Seed | `0` |
| 验收日期 | 2026-09-11 |

Checkpoint SHA256：

```text
cf2bc864900ce38b85daa68300addb8fdef79cc7102bd652a99f5f1924b69346
```

本次验收采用手册的**判据 A**：使用**同一个官方免训练 checkpoint**、**同一批 20 条序列**、**同一 seed**，分别通过 **CALVIN 官方 evaluator** 与 **Forge 两节点链路**执行，并比较 benchmark 成绩。

### 公开基线对比（debug 数据：官方 evaluator vs Forge）与复现

"基线"指**在官方 debug 数据上、由 CALVIN 官方 evaluator 跑出的结果**，Forge 与之对比。

复现步骤：

**Step 0 下载并校验**
```bash
wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
sha256sum calvin_debug_dataset.zip          # 期望 c66d0914…（官方 debug 数据集）
wget http://calvin.cs.uni-freiburg.de/model_weights/D_D_static_rgb_baseline.zip
# 解压后 mcil_baseline.ckpt 的 sha256 = cf2bc864…
```

**Step 1 修正 debug 的语言嵌入变体（关键）**
- **原因**：debug 包的 `validation/lang_annotations/embeddings.npy` 是 **`paraphrase-MiniLM-L3-v2`**（md5 `ab729fc0…`），而该 checkpoint 训练时用的是 **`paraphrase-MiniLM-L6-v2`**：
  - CALVIN 代码默认 `calvin_models/conf/model/sbert.yaml = paraphrase-MiniLM-L6-v2`；
  - git 历史：2023-02 前配置写 `mini`，而 `language_network.py` 中 `mini` 解析为 `paraphrase-MiniLM-L6-v2`；checkpoint 为 2022-02-17（属"mini=L6"时代）。
  - 若不修正，官方 MCIL 的 LH-MTLC SR1 塌到 ≈0.06（嵌入空间不匹配，策略近似忽视指令）。
- **做法（只改仓库外的数据副本，不改 checkpoint/benchmark/CALVIN）**：使用**官方 `task_D_D/validation` 的原始元数据**，通过 HTTP Range 仅取小文件（`fetch_zip_members.py`）：
  - `lang_annotations/embeddings.npy`（md5 `b0755f42…`，与用 `paraphrase-MiniLM-L6-v2` 重算结果 `max_abs_diff=0.0`）；
  - `lang_annotations/auto_lang_ann.npy`（md5 `f25507e2…`）；
  - `.hydra/merged_config.yaml`、`statistics.yaml`（与 debug 包**逐字节相同**）。
  将上述放入数据副本（如 `calvin_acceptance/data/calvin_debug_dataset`），供两侧共同使用。

**Step 2 Official 侧（只调用官方函数，不改 CALVIN）**
```bash
N=20 OUT=official_repr20.json python official_subset.py
# 内部：seed_everything(0) → get_default_model_and_env + get_sequences(1000)[:20] + evaluate_sequence
```

**Step 3 Forge 侧（两节点）**——`config.yaml`：`sequence_count: 1000`、`task_ids: "0-19"`、`dataset_path` 指向上述数据副本；配 validation-only `real_policy.py`：
```bash
dora run dataflow.policy.yaml
```

**Step 4 逐序列比较（Wilson 95% CI + Fisher 精确检验）**
```bash
python compare_acceptance.py --official official_repr20.json \
  --forge <forge_result>.json --out acceptance_comparison_repr20.json
```

### 成绩对照

| 路径 | SR1（成功/总数） | avg_seq_len | SR2 | 逐序列一致 | 95% Wilson（SR1） |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 官方 evaluator（debug，`get_sequences(1000)[:20]`） | **7 / 20 = 0.35** | **0.35** | 0.00 | — | `[0.181, 0.567]` |
| Forge Benchmark Node（debug，同 20 条） | **7 / 20 = 0.35** | **0.35** | 0.00 | **20 / 20** | `[0.181, 0.567]` |
| **差异** | **0** | **0.00** | 0 | — | **Fisher exact p = 1.000**（CI 重合） |

逐序列 `chain_length`（两侧相同）：

```text
[0,1,0,0,1,0,1,1,1,0,1,0,0,1,0,0,0,0,0,0]
```

补充结果：

- 两边 95% Wilson 置信区间重叠；
- Fisher 双侧精确检验 `p = 1.000 > 0.05`；
- Official 与 Forge 均执行同一批 `get_sequences(1000)[:20]`，无重复、无缺失；
- checkpoint、协议、robot/action type、seed、episode 数保持一致。

### 判定（LH-MTLC验收实验）

| 条件 | 结果 |
| :--- | :--- |
| 官方 baseline 为明确非零成绩 | ✅（SR1 = 7/20 = 0.35，非零） |
| 两节点 `dora run` 跑通 | ✅ |
| 同 checkpoint | ✅ |
| 同协议/序列集 | ✅ |
| 同 episode 数 | ✅ |
| 同一批序列 | ✅ |
| 逐序列结果一致 | ✅（20 / 20） |
| 两边 Wilson 95% CI 重叠 | ✅ |
| Fisher exact `p > 0.05` | ✅ |
| Forge 成绩与官方处于同一统计区间 | ✅ |
| **最终判定** | **✅ 通过，判据 A** |

**结论：CALVIN Benchmark Node 通过正式验收（判据 A）。** Official 与 Forge 在 20 条序列上逐序列完全一致（SR1 均 7/20，`avg_seq_len` 均 0.35），Wilson 区间重合，Fisher `p = 1.0`，没有观察到统计上可区分的差异。因此现有证据支持：**Forge 两节点执行链路没有系统性改变 CALVIN LH-MTLC 的任务、场景、观测、动作与评分语义。**

### 外部审查（交叉验证）问题回应（精简）

| 审查问题 | 结论 | 依据 |
| :--- | :--- | :--- |
| P1-1 N=20 统计功效不足 | 已由 N=50 回应 | 已有 official/Forge 各 50：逐序列 **50/50** 一致、Fisher p=1.0（`evidence/acceptance_comparison_L6.json`）。逐序列一致强于统计非劣。 |
| P1-2 复现材料不入库 | 不处理（合理） | harness 为 validation-only、非交付物；记录含完整复现步骤，手册未要求其入库。 |
| P1-3 三态终止偏离 | **已修复** | 子任务预算耗尽现记 `timed_out`；`chain_length`/`num_steps` 不变。 |
| P1-4 policy 侧 RNG 前置 | 非接入错误，仅记前置 | benchmark 节点逐位确定；随机性属 MCIL。验证策略已按官方 `seed_everything(0)`（模型加载后、首次 step 前）。 |
| P2-4 policy 侧门控前置 | 非 benchmark 缺陷 | dora 跨通道乱序是框架性质，手册 §3.1 已列已知、watchdog 为待办；修复需动共享 `_core`（硬约束禁止）。前置条件已写入 README 与 `dataflow.example.yaml`。 |
| P2-5 MTLC 无端到端 | **已补做** | 见「补充：MTLC 验收实验」：22/22 逐 trial 一致 + 全量映射算法等价。 |
| P2-6 靶子/全量 | 不处理（合理） | 全量 task_D_D 177GB 不可得；leaderboard 数字基于全量验证集，与 debug 20/50 不可比。 |
| P2-7 文档矛盾 | 已修正+说明 | `robot_obs` 逐位相同、`scene_obs` 存在 ~0.002 物理残差（未覆盖速度自由度）；图像不逐位属渲染器固有（两个独立官方 env 同样不一致）；不测 CPU/GPU 峰值不影响正确性。 |


### 补充：MTLC 验收实验（LH-MTLC验收实验在上面）

- **全量索引映射等价**：取官方 `task_D_D/validation/ep_start_end_ids.npy`（4 episodes），用官方 `DiskDataset(key="vis")` 的 `episode_lookup` 与 adapter `_episode_lookup_from_bounds` 比对：长度均 98958、**0 mismatch**，全部 334 个 checkpoint id 有效（max 97777）。
- **逐 trial 对照**：因全量 task_D_D（177GB）不在本机，用过滤到 debug 有效 id 的 checkpoint 副本（12 任务 / 22 trial），官方 `evaluate_policy_singlestep` 的 `rollout` 与 Forge MTLC 两节点同数据对照：**22/22 逐 trial 一致**（task/dataset_id/solved），两侧均 3/22 成功。证据：`evidence/official_mtlc.json`、`evidence/forge_mtlc/*.json`、`evidence/mtlc_comparison.json`。

### 最终候选代码 Runtime Smoke

最终候选代码（refactor 后）额外执行了一次端到端对照（`get_sequences(1000)[:5]`）：Official 与 Forge 的 `chain_length` 均为 `[0,1,3,0,1]`，**逐序列 5/5 一致，`avg=1.0`**。同时 stub 两节点冒烟：观测数 == 动作数（`obs == act`），结果 JSON 正常落盘，MCIL 权重与 384 维语言嵌入加载正常，无 traceback/`KeyError`。

### Deterministic Trace 对照

固定动作、同序列，官方 env 路径 vs Forge adapter：

- `robot_obs`(15D)、TaskOracle：**逐帧逐位相同**；
- `scene_obs`(24D)：official ↔ Forge **逐帧一致**（0 mismatch）；
- 两路图像：跨 env 实例不一致（两个独立官方 env 亦复现；属 pybullet/EGL 渲染器跨 client 不确定性），不影响判分。

**反向故障注入（negative control）** `BAD_ACTION_COMPONENT=0`、`BAD_ACTION_DELTA=0.25`：

| 比较 | robot max abs diff | 首个不符帧 |
| :--- | ---: | ---: |
| 正向（同动作、两次独立 reset） | **0.0（逐位相同）** | — |
| 反向（注入扰动） | **6.24** | **第 0 帧** |

说明：扰动一个动作分量即导致状态显著偏离；正向时 `robot_obs` 逐位相同。

**备注项（均不影响判据 A，且非接入错误）**

1. **`scene_obs` 的 ~0.002 残差**：同一 env 反复 reset 时，`env.reset(robot_obs, scene_obs)` 只恢复这两个数组、未覆盖速度等自由度，两次独立 reset 间可有 ~0.002 物理残差（`deterministic_trace_negative_control.json`：`forward_robot_maxdiff=0.0`、`forward_scene_maxdiff=0.002`）。属 CALVIN 环境 init-state 覆盖度问题（手册 §7 陷阱 1 同类）；official 与 Forge 的 fresh-env trace 中 `robot_obs`/`scene_obs` 均 0 mismatch，TaskOracle 不受影响。**不可改**：修物理不属 benchmark 接入职责，改渲染/物理也非正解。
2. **图像 bit-level 不一致**：pybullet/EGL 跨 client 渲染固有，两个独立官方 env 同样不一致（RoboDojo 记录的 Isaac Sim 渲染亦同此现象）。故按**状态层面对齐**，不强制图像逐位。**不可改**：属渲染器性质，非 benchmark 接口语义。

### 接入过程中发现并修复的问题

1. **debug 包语言嵌入变体错误**：`lang_annotations/embeddings.npy` 为 L3，checkpoint 用 L6 → 改用官方 `task_D_D` 原始元数据（HTTP Range 取回，md5 校验；与重算 L6 逐位一致）。
2. **env 构建崩溃**：默认会实例化未使用的 tactile 相机，在 py3.12 触发 `networkx==2.2` 报错 → 只构建实际发布的 `rgb_static`/`rgb_gripper`。
3. **契约测试未同步**：`structure-fixed` 版要求显式 `suite`/`policy_id`，旧测试未更新 → 对齐测试。
4. **dora 跨通道乱序**：首帧观测可能先于 `set_instruction`/`reset_scene` 到达，策略丢弃首帧致静默悬停 → policy 端按"指令+reset 门控"处理。
5. **缺启动观测自检日志**：未打印分辨率/语义 → 构造时打印 observation contract（并写入 `reproducibility`）。
6. **`close()` 未吞异常**：→ 加 `try/except`。
7. **adapter 代码清理 R1–R6**（行为无损）：兼容层集中、观测契约去重、`step` 拆 `_lh_step`/`_mtlc_step`、`MtlcProtocol` 状态机化、删除死代码、分区/常量。
8. **三态终止修复**：子任务预算耗尽此前被共享核心记为 `partial`（`_core` 无 timed_out 通道）；在 adapter 翻译层发布 `cfg.max_steps` 截止步使 `timed_out` 精确触发，计分不变。

### 最终代码版本

```text
benchmark-calvin:
  a9cd643  pristine: calvin benchmark selfcontained v14.1-structure-fixed
  c9a7b51  fix: env 只构建实际发布相机 + 合约测试对齐
  9837685  fix: 启动观测自检 + close 吞异常
  c0720d2  refactor: behavior-preserving cleanup (R1-R6)
  tags: pristine-v14.1 / pre-fix / post-fix / refactor-base / refactor-post
benchmark-libero: e01cd66（shared _core 来源）
CALVIN:           fa03f01（calvin_env 1431a46）
```

> 追加改动（commit `ccecba1`）：`adapters/calvin_adapter.py`（三态终止修复）、
> `tests/test_termination_contract.py`（新增）、`tests/test_review_contract.py`（计数器初始化）、
> `README.md` / `dataflow.example.yaml`（文档）。共享 `_core/**`、`bench_tool/**`、`result.py` 未动。

Pre-commit gate（三态终止修复后重跑）：

```text
git diff --check                     PASS
py_compile                           PASS
pytest                               PASS (86 passed；新增 6 条终止契约测试)
受保护文件逐字节一致（_core/bench_tool/result.py vs benchmark-libero）  PASS
diff 范围审计（无越界改动）           PASS（仅 adapter 翻译层 + 测试）
计分函数字节码审计（g1）              PASS（仅 4 个终止管道 code object 变化）
差分等价（旧 adapter vs 新 adapter，LH+MTLC）                         PASS
同 session Official vs Forge（get_sequences(1000)[:5]）               PASS (5/5, [0,1,3,0,1])
MTLC 端到端（official vs Forge，22 trial）                            PASS (22/22)
全量 MTLC 索引映射等价（334 id）                                      PASS (0 mismatch)
```

checkpoint / 配置 / commit 已记录；结果 JSON 含 `mean_policy_latency_ms`（按 RoboDojo 口径，不额外测 CPU/GPU 峰值）。


### 最终结论

**CALVIN Benchmark Node（LH-MTLC）：✅ 通过，判据 A。**

```text
同 checkpoint (cf2bc864…)、同 get_sequences(1000)[:20]、同 seed 0

Official : avg_seq_len = 0.35, SR1 = 7/20 = 0.35, hist {0:13, 1:7}
Forge    : avg_seq_len = 0.35, SR1 = 7/20 = 0.35, hist {0:13, 1:7}
逐序列一致 = 20 / 20
Fisher exact p = 1.000, Wilson 95% CI overlap = true
```

在当前 20-episode、同 checkpoint、同序列集合的测试条件下，Forge 两节点执行链路与官方 evaluator **逐序列一致**，满足 Benchmark 接入手册 v14.1 的判据 A。

**补充（修复后）**：LH-MTLC 三态终止已修复为 `timed_out`（计分不变，经字节码审计 + 修复前后差分 + 同 session N=5 复验证明未破坏判据 A）；**MTLC 端到端已补做并 22/22 逐 trial 一致**。两条协议均满足手册 §3.4 / §8。

### 关于三态终止修复：不是修改官方语义，而是契约层级不匹配下的必要适配

1. **官方语义**：CALVIN 官方评测没有 `success/failed/timed_out` 概念，只记录 `chain_length`；其终止规则是"某个子任务在 360/240 步预算内未完成，即结束整条链"。这是**子任务预算**，与手册 §3.7 的"单 episode 步数上限" `max_steps` 不是同一层级。
2. **Forge 契约**：手册 §3.4/§8 要求 episode 终止态区分 `success / failed / timed_out`，因此"子任务预算耗尽"必须映射为 `timed_out`，否则三态无法区分。
3. **冲突不在 adapter**：共享 `_core` 的 runner 只在 `step_idx >= max_steps`（episode 级）产出 `timed_out`，其余 `done && !success` 一律记 `partial`；且 `StepOutcome` 无"终止原因"字段，`_core/**` 须与 benchmark-libero 逐字节一致、禁止修改。即**共享核心没有给 adapter 提供表达"子任务超时"的通道**。
4. **因此 adapter 的做法是合理的**：由 adapter 自行维护**累计动作计数与当前子任务截止步**，每步通过 `cfg.max_steps` 将该截止步传给冻结 runner，使 runner 恰在官方预算边界进入其**已有**的 `timed_out` 分支。这是在"不改共享核心、不改官方评测"前提下，**唯一能正确区分三态的方式**。
5. **无副作用**：观测、动作、步数、`chain_length`、`success` 全部不变（已由字节码审计、修复前后差分、同 session official vs Forge N=5 验证），仅修正 Forge 层终止标签。

**结论**：这不是接入错误，也不改变官方语义；是 CALVIN 特有"子任务超时"与 Forge 三态契约之间的层级差异所导致的、不可避免且必要的适配。
