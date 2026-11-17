# NMFlib

纯 MoonBit 实现的非负矩阵分解（NMF）库。`V ≈ W·H`，W、H 非负。

> 状态：v0.1.0 开发中。**T01–T14 已完成**：MU-Frobenius、CD-Frobenius、NNDSVD/NNDSVDa 初始化、MU-KL、L1/L2 正则、
> Hoyer 稀疏度指标与属性/Fuzz 测试体系可用，188 项测试在 wasm / wasm-gc / js / native 四后端全部通过。

[![CI](https://github.com/L1r0gn/NMFlib/actions/workflows/ci.yml/badge.svg)](https://github.com/L1r0gn/NMFlib/actions/workflows/ci.yml)

仓库地址：<https://github.com/L1r0gn/NMFlib>

## 功能支持情况

| 功能 | 状态 |
|---|---|
| MU-Frobenius 求解器（Lee & Seung 乘法更新，诚实收敛标记） | ✅ 已实现 |
| Random 初始化（SplitMix64，固定 seed 逐位可复现） | ✅ 已实现 |
| fit / fit_transform / transform / inverse_transform / reconstruction_error | ✅ 已实现 |
| 完整错误处理（`Result` + 7 类错误枚举，不静默修改输入） | ✅ 已实现 |
| 可运行示例 `examples/quickstart` | ✅ 已实现 |
| CD-Frobenius 求解器（循环坐标下降，精确坐标步） | ✅ 已实现（T08） |
| 内部截断 SVD（幂迭代 + deflation，仅供 NNDSVD，不公开） | ✅ 已实现（T09） |
| NNDSVD 初始化（标准正负部选对，确定性） | ✅ 已实现（T09） |
| NNDSVDa 初始化（NNDSVD + 零元素全局均值填充，确定性） | ✅ 已实现（T10） |
| 广义 KL 散度（Generalized KL，MU 求解器） | ✅ 已实现（T11） |
| MU-KL 求解器（乘法更新，零值/零行列有明确策略） | ✅ 已实现（T11） |
| MU-KL 的 transform（固定 H，KL 更新 W） | ✅ 已实现（T11） |
| L1/L2 正则（进入目标函数、更新公式与收敛判断，三种求解组合均支持） | ✅ 已实现（T12） |
| Hoyer 稀疏度指标（`sparsity`，边界约定见下文） | ✅ 已实现（T13） |
| 属性测试 + 轻量 Fuzz（固定 seed：200 组合法 + 244 组非法） | ✅ 已实现（T14） |
| Topic Modeling 应用示例 `examples/topic_modeling`（主题建模 MVP） | ✅ 已实现（T17） |

选择不支持的 CD+KL 组合，或传入负数 / NaN / Infinity 的正则参数时，`fit` 返回 `Err(InvalidConfig)`，不会静默切换求解器、损失，也不会把非法参数替换为 0。

### Solver × Loss 兼容矩阵

| Solver | Frobenius | KL |
|---|---|---|
| MU | ✅ 已实现 | ✅ 已实现（T11） |
| CD | ✅ 已实现 | ❌ `InvalidConfig` |

L1/L2 正则对 MU+Frobenius、CD+Frobenius、MU+KL 三种组合均生效；CD+KL 仍为 `InvalidConfig`。

纯 MoonBit、零第三方依赖（仅 core），wasm / wasm-gc / js / native 四后端均执行真实测试。

## 安装

发布到 mooncakes.io 后：

```sh
moon add L1r0gn/nmf
```

尚未发布到 mooncakes.io，当前可直接克隆源码使用：

```sh
git clone https://github.com/L1r0gn/NMFlib.git
cd NMFlib
moon run examples/quickstart
```

## 快速上手

与 `examples/quickstart/main.mbt` 一致（已实际运行验证）：

```moonbit
fn demo() -> Result[Unit, @nmf.NMFError] {
  // 1. 构造非负矩阵 V（3×3，行主序）
  let v = match @nmf.Matrix::new(3, 3, [1.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 6.0]) {
    Ok(m) => m
    Err(e) => return Err(e)
  }

  // 2. 配置并训练（MU + Frobenius + Random，固定 seed 可复现）
  let config = @nmf.NMFConfig::new(n_components=2, seed=42UL)
  let model = @nmf.NMF::new(config)
  match model.fit(v) {
    Ok(_) => ()
    Err(e) => return Err(e)
  }

  // 3. 取因子、计算重构误差 ‖V − WH‖_F
  let w = match model.w() { Ok(m) => m; Err(e) => return Err(e) }  // 3×2
  let h = match model.h() { Ok(m) => m; Err(e) => return Err(e) }  // 2×3
  let err = match model.reconstruction_error(v) {
    Ok(x) => x
    Err(e) => return Err(e)
  }
  println("W: \{w.rows()}x\{w.cols()}, H: \{h.rows()}x\{h.cols()}, error=\{err}")
  Ok(())
}

fn main {
  match demo() {
    Ok(_) => ()
    Err(e) => println("NMF failed: \{e}")
  }
}
```

在仓库根目录运行：

```sh
moon run examples/quickstart
```

实际输出（seed=42，两个工具链版本一致）：

```text
W: 3x2, H: 2x3
iterations=64, converged=true
reconstruction error = 1.8065620213588327
```

### 应用示例：Topic Modeling（T17）

`examples/topic_modeling` 是可运行的主题建模 MVP：固定语料 → 空白 tokenizer →
文档—词频矩阵 → MU + 广义 KL + Random 分解 → 主题关键词 / Hoyer 稀疏度 /
文档主题激活 → 新文档 `transform` 推理，全程带运行时自检。详见
[examples/topic_modeling/README.md](examples/topic_modeling/README.md)。

```sh
moon run examples/topic_modeling
```

T17 验证期间官方工具链发布 0.1.20261117，触发 CI 的 fail-closed 版本断言；已按
`scripts/ci-install-moonbit.sh` 内置流程完成本地全量验证（check --deny-warn +
四后端 188 项测试 + 两个示例，输出与 0.1.20261028 逐字一致）后更新 CI pin。
T17 提交 `740c622` 的 run #20（2026-11-17，分支 `feat/t17-topic-modeling`），
5 个 Job 实例全部成功；Unit & Integration Tests job 真实运行了两个示例，
topic_modeling 的全部运行时自检在 CI 中通过。

### NMFConfig 使用方法

全部参数为标签参数，除 `n_components` 外均有默认值：

```moonbit
let config = @nmf.NMFConfig::new(
  n_components=2,            // 分解秩 k，必填，1 ≤ k ≤ min(rows, cols)
  solver=@nmf.Solver::MU,    // 默认 MU；可改为 CD（循环坐标下降，仅支持 Frobenius）
  loss=@nmf.Loss::Frobenius, // 默认 Frobenius；MU 可改为 KLDivergence（CD 不支持 KL）
  init=@nmf.Init::Random,    // 默认 Random；可改为 NNDSVD / NNDSVDa（确定性）
  max_iter=500,              // 默认 200；耗尽未达标时 converged=false
  tol=1.0e-6,                // 默认 1e-4；相邻目标相对变化 |ΔF|/max(1,|F|) < tol 即收敛
  seed=2026UL,               // 默认 42；同 seed 结果逐位一致
  l1_reg=0.0,                // 默认 0；L1 强度，同时作用于 W、H，必须有限且 >= 0
  l2_reg=0.0,                // 默认 0；L2 强度，同时作用于 W、H，必须有限且 >= 0
)
```

训练结果通过 `model.w()` / `model.h()`（防御性拷贝）、`model.n_iter()`、
`model.loss_value()`（完整优化目标，含正则项）、`model.converged()` 读取；
`model.reconstruction_error(v)` 始终返回范数 ‖V−WH‖_F（不含正则项；零正则时等于 sqrt(2·loss)）。
未训练的模型调用以上接口一律返回 `Err(NotFitted)`；fit 失败会回到未训练状态。
`model.transform(x)` 固定 H 对新数据解 W；`model.inverse_transform(w)` 计算 W·H。
`solver=@nmf.Solver::CD` 时 fit 与 transform 都使用 CD 坐标下降，不会回退到 MU。
`init=@nmf.Init::NNDSVD` 使用内部幂迭代截断 SVD；初始化确定性、不依赖 `seed`，只影响
fit 的初始 W/H（transform 仍沿用既有行为，用 Random 初始化新数据的 W）。内部 SVD 不是
公共 API。`init=@nmf.Init::NNDSVDa` 在标准 NNDSVD 之后，把 W/H 中**精确零元素**替换为
原始 V 的全局均值 `mean(V)`；非零元素保持不变。`mean(V)=0`（全零输入）时保持全零，
不引入随机噪声。

### NNDSVDa 配置示例

```moonbit
let config = @nmf.NMFConfig::new(
  n_components=2,
  init=@nmf.Init::NNDSVDa,
  seed=42UL, // NNDSVDa 确定性，seed 不影响初始化
)
```

与 NNDSVD 一样，NNDSVDa 只影响 fit 的初始 W/H；transform 固定训练 H，使用既有 Random
约定初始化新数据的 W。

### MU-KL（广义 KL 散度）

MU 求解器支持广义 KL 目标（不是归一化概率分布 KL，不要求元素和为 1）：

```text
D_KL(V ‖ WH) = Σ [ V·log(V/WH) − V + WH ]
```

零值约定：

- `V=0, WH=0`：该项为 0，不计算 0/0；
- `V=0, WH>0`：对数项按极限为 0，该项为 WH；
- `V>0, WH=0`：散度为 +∞；`fit`/`kl_divergence` 返回 `NumericalError`，不伪造有限损失。

实现使用 `V(log V − log WH) − V + WH` 代替 `V/WH`，避免比值溢出；对累计和使用补偿求和。
不使用全局 epsilon 下限：更新分母为零时，对应 H 行/W 列对 WH 无影响，规范化为 0；
非零分母精确使用，非有限值明确报错。KL 模型：

- `model.loss_value()` 返回真实 KL 散度；
- `model.reconstruction_error(v)` 仍返回 Frobenius 范数 `‖V−WH‖_F`（向后兼容）；
- `model.transform(x)` 固定训练 H，用 KL 的 W 更新公式。

```moonbit
let config = @nmf.NMFConfig::new(
  n_components=2,
  solver=@nmf.Solver::MU,
  loss=@nmf.Loss::KLDivergence,
  seed=42UL,
)
```

NNDSVD/NNDSVDa 可用于 MU-KL，但公共初始化结果不被修改，也不静默替换 Init；
如果初始 WH 在 V>0 处为零，MU-KL 明确返回 `NumericalError`。

### L1/L2 正则化

正则项同时作用于 W 和 H：

```text
Ω(W,H) = l1_reg·(ΣW + ΣH) + (l2_reg/2)·(‖W‖²_F + ‖H‖²_F)

J_F (W,H)  = ½‖V − WH‖²_F + Ω(W,H)
J_KL(W,H)  = D_KL(V‖WH)  + Ω(W,H)
```

因为 W、H 非负，`ΣW = ‖W‖₁`、`ΣH = ‖H‖₁`，第一项就是两个因子的 L1 范数之和。
正则**以梯度形式进入更新公式**，不是训练结束后对因子做截断或缩放：

| 求解器 | H 更新（W 更新对称） |
|---|---|
| MU-Frobenius | `H ← H ⊙ (WᵀV) ⊘ (WᵀWH + l1_reg + l2_reg·H)` |
| CD-Frobenius | 曲率 `G_pp + l2_reg`，梯度 `(G·H)_p − C_pj + l1_reg + l2_reg·H_p`，`H_p ← max(0, H_p − g/曲率)` |
| MU-KL | `H ← H ⊙ (Wᵀ(V⊘WH)) ⊘ (colsum(W) + l1_reg + l2_reg·H)` |

语义约定：

- `model.loss_value()` 返回**完整目标** `J_F` 或 `J_KL`（含正则项），收敛判断也用同一个量；
- `model.reconstruction_error(v)` 始终是 `‖V − WH‖_F`，**不含**正则项；
- `model.transform(x)` 固定训练得到的 H，只对正在求解的 W 施加正则（H 的惩罚项对 W 是常数，
  不改变 W 的更新，但计入收敛判断监视的完整目标）；
- `l1_reg = l2_reg = 0`（默认）逐位复现 T11 之前的算法与结果：加入的 `0.0` / `0.0·x`
  项在 IEEE-754 下精确，不改变任何数值；
- 负数、NaN、±Infinity 一律 `InvalidConfig`，不会被静默替换为 0。

```moonbit
let config = @nmf.NMFConfig::new(
  n_components=2,
  l1_reg=0.1,   // L1：推动因子稀疏
  l2_reg=0.2,   // L2：抑制因子幅度
  seed=42UL,
)
```

CD+KL 组合不受影响，仍然返回 `InvalidConfig`；三个有效组合（MU+Frobenius、
CD+Frobenius、MU+KL）均支持任意 `l1_reg` / `l2_reg` 组合。

### Hoyer 稀疏度

`@nmf.sparsity(x)` 计算单个非负向量的 Hoyer 稀疏度（Hoyer, JMLR 2004）：

```text
S(x) = (√n − ‖x‖₁/‖x‖₂) / (√n − 1)
```

取值范围 `[0, 1]`：所有元素相等 → `0`（完全不稀疏）；只有一个非零元素 → `1`（最大稀疏）。
指标对正数缩放（`S(cx) = S(x)`，`c > 0`）和元素排列都不变。

```moonbit
let s = @nmf.sparsity([1.0, 1.0, 0.0, 0.0]) // Ok(0.5857864376269051)
let t = @nmf.sparsity([1.0, 0.0, 0.0, 0.0]) // Ok(1.0)
let u = @nmf.sparsity([2.0, 2.0, 2.0])       // Ok(0.0)
```

边界约定（按此顺序判定）：

| 输入 | 结果 |
|---|---|
| 空向量 `[]` | `Err(InvalidConfig)`：公式无定义 |
| 单元素 `[x]` | `Ok(0.0)`：分母 `√n−1 = 0`，约定为 0（优先于全零规则，故 `[0.0] → 0.0`） |
| 全零向量（`n ≥ 2`） | `Ok(1.0)`：约定"没有非零分量即最大稀疏"（公式为 0/0） |
| 含负元素 | `Err(NegativeEntry(0, idx))`：向量按 1×n 行处理，第一维恒为 0 |
| NaN / ±Infinity | `Err(NumericalError)`：先于负值判断，故 `-Infinity` 也报 NumericalError |

数值策略：先除以最大元素得到 `y ∈ [0,1]` 再求和（`Σy ≤ n`、`Σy² ≤ n`，不会像直接算
`Σx²` 那样在大数处上溢、在次正规数处下溢），两个和都用 Kahan 补偿；仅在 `|S| ≤ 1e-12`
的舍入残差内夹到 `0`（等元素向量会出现这种情况），超出容差说明计算有缺陷，返回
`NumericalError` 而不是静默截断。

**统计方向**：`sparsity` 只接受单个向量，不提供矩阵级平均——因为矩阵级聚合必须替调用方
选择方向与零向量规则。NMF 中两个因子有各自的自然方向，按需提取后逐个计算：

- `W`（`m×k`）的**列**是基向量，长度 `m`；
- `H`（`k×n`）的**行**是成分的样本激活向量，长度 `n`。

```moonbit
// 每个基向量的稀疏度（W 的列）
let w = match model.w() { Ok(m) => m; Err(e) => return Err(e) }
for j in 0..<w.cols() {
  let column = Array::make(w.rows(), 0.0)
  for i in 0..<w.rows() {
    column[i] = match w.get(i, j) { Ok(x) => x; Err(e) => return Err(e) }
  }
  println("W[:, \{j}] sparsity = \{@nmf.sparsity(column)}")
}
```

把整个矩阵展平后计算不是 W 或 H 的稀疏度，也不等于各成分稀疏度的平均。指标是纯函数，
不读取也不修改模型状态。

## 运行测试

```sh
moon check --deny-warn   # 类型与警告检查
moon test                # 全部 209 项测试（默认 wasm 后端；188 项原有 + 21 项 T16 sklearn 参考 fixtures）
moon test --target js    # JS 后端（真实执行全部测试）
moon test --target wasm-gc
moon test --target native
moon fmt --check         # 格式检查
```

测试分层：Matrix 单元/白盒内核 → Error 行为 → SplitMix64/Random → Frobenius 损失 →
广义 KL 损失（手算/零值/非负性/溢出）→ MU 求解器（Frobenius + KL；KL 单轮更新、目标不增、
零行列、初始化兼容）→ CD 求解器（精确坐标步、目标不增、零曲率/零行列/极小值边界）→
内部 SVD（幂迭代/deflation/秩不足/重复奇异值/极小值）→ NNDSVD/NNDSVDa（正负部选对、
零退化、非负、确定性、零元素均值填充、全零保持）→ L1/L2 正则（手算目标值、三求解器
单轮更新对照独立参考实现、完整目标单调性、强 L1 收缩、transform 只正则 W、零正则逐位兼容、
参数合法性）→ Hoyer 稀疏度（手算样例、尺度/排列不变性、零向量与维度边界、非法输入、
极大极小量级、非方阵行/列方向、范围性质）→ NMF API → 端到端验收（普通矩阵、已知低秩、
全零、非法输入、seed 复现、未收敛状态）+ 200 组随机属性扫描。CD、NNDSVD、NNDSVDa、KL、
MU-KL、正则与稀疏度指标均有独立测试。

### 属性测试与轻量 Fuzz（T14）

`src/property_test.mbt` 与 `src/fuzz_test.mbt` 使用项目自带的 `TestRng`（测试专用 LCG，
独立于库内 SplitMix64），**seed 固定、无时钟、无外部随机源**，同 seed 在任一后端生成相同输入。

五项性质与断言方式：

| 性质 | 断言 |
|---|---|
| W/H 非负 | 逐元素 `x >= 0` 且有限 |
| 维度正确 | `W: m×k`、`H: k×d` 严格相等 |
| 同 seed 可复现 | 同 config 两次 `fit` 的 `W/H/loss/n_iter` 位相等 |
| 合法输入不 panic | 每次调用返回 `Ok`，后续断言可达（MoonBit 中 panic 会终止测试） |
| 非法输入返回 Err | 错误变体精确匹配（见下） |

规模与 seed：

| 测试 | seed | 用例数 |
|---|---|---|
| 合法输入五性质（含 fit + transform + sparsity） | `20261116` | 200 组（fit 200 次、transform 200 次） |
| 确定性初始化 + MU-KL（全正数据） | `20261117` | 12 组 |
| 目标单调性（三求解器 × 5 个预算前缀） | `20261118` | 8 组输入 × 3 组合 × 5 预算 |
| sparsity 范围/尺度/排列不变性 | `20261119` | 100 组向量 |
| 非法 fit / CD+KL / transform / 未训练 / 构造边界 / 状态安全 / sparsity | `20261116` + 偏移 | 80 + 16 + 60 + 40 + 12 + 6 + 30 = **244** |

生成策略：尺寸 2–9（非方阵优先）、rank 1–min(m,d)、75% 含 `(i+j)%3` 精确零元素、
10% 量级 1e-3 / 1e3、三种初始化、三组合、每 5 例开启 L1/L2 正则、预算 5–34。
失败信息形如
`property failed: seed=20261116 case=0 W[0,0] negative: -1.0`，
可直接用 seed + case 复现。尺寸有界（≤12/边），超大形状只通过构造函数的溢出校验测试，
不会申请不受控内存。

范围说明：

- `fit`/`transform`/`reconstruction_error` 只接受 `Matrix`，而 `Matrix::new` 已拒绝
  NaN/±Inf，因此这些值**无法**到达模型 API；Fuzz 在构造边界与实际接受 `Array[Double]` 的
  `sparsity` 上覆盖它们（不虚构不存在的错误路径）。
- MU-KL 与 NNDSVD/NNDSVDa 组合在结构零处可能返回文档化的 `NumericalError`，故主扫描用
  Random 初始化，确定性初始化单独用全正数据覆盖。
- 已训练模型再次 `fit` 失败时，按既有契约回到**未训练**状态（不是保留旧模型），Fuzz 对该契约断言。

CI（`.github/workflows/ci.yml`）在 push / PR 时运行：格式与 check、完整测试套件、
三后端 check+test（matrix 展开为 native / js / wasm-gc），以及独立的轻量
`Python Reference Smoke` Job（T16 参考工具自测 + 单数据集单 seed 小型对照，
完整多 seed 实验不进 CI，按需本地运行）。
工具链版本采用 fail-closed 固定策略（见 `scripts/ci-install-moonbit.sh`）：官方安装脚本安装后
断言版本号，不符即失败。

CI 已在 GitHub Actions 实际运行并通过：T14 提交 `61a32ec` 的 run #11（2026-11-16），
5 个 Job 实例（MoonBit Check、Unit & Integration Tests、Backend native / js / wasm-gc）全部成功；
T16 提交 `9d8a0f3` 的 run #17（2026-11-17，分支 `feat/t16-sklearn-reference`），
6 个 Job（上述 5 个 + Python Reference Smoke）全部成功。
上述命令亦已在本地按 CI 相同配置验证通过（本地工具链 0.1.20261028；CI 锁定 0.1.20261104），
209 项测试在四后端各执行一遍。

## 数值参考验证（T16：scikit-learn 对照）

`validation/sklearn_reference/` 是独立的 Python 参考验证工具（**不是生产依赖**，
未向 MoonBit 包引入任何 Python/sklearn 依赖，也未为对照修改任何公开 API）：

- 以 sklearn 1.7.2 `decomposition.NMF` 为独立参考，对 MU+Frobenius、CD+Frobenius、
  MU+KL 三个有效组合做系统对照；CD+KL 仍为 `InvalidConfig`，不参与对照。
- 双方最终 W/H 一律用独立的 `objective.py` 重新计算完整目标
  `J = D(V‖WH) + λ1(ΣW+ΣH) + (λ2/2)(‖W‖²_F+‖H‖²_F)`，不信任任何一侧自报 loss。
- 正则参数映射 `r=λ1/(λ1+λ2)`、`alpha_W=(λ1+λ2)/n_features`、
  `alpha_H=(λ1+λ2)/n_samples`（alpha_H 永远显式设置，不用默认 `"same"`），
  经手算验证（L1 only / L2 only / L1+L2 / 关闭 × 非方阵）。
- 4 个固定 seed 数据集：低秩小/中、带噪声低秩、稀疏计数（**真实 rank 2 < 配置 k=4**
  的秩不足矩阵，含 35.8% 精确零）；主网格 5 seeds × 4 正则配置 × 3 组合 +
  NNDSVD/NNDSVDa 确定性层，共 336 案例 × 2 实现 = 672 次 fit。
- 明确记录两侧差异：收敛判据（NMFlib 每轮完整目标相对变化；sklearn CD 用 KKT
  violation 比、MU 每 10 轮用无正则误差相对变化）、MU 更新顺序、sklearn MU-KL 的
  `EPSILON=1.19e-7` 分母保护 vs NMFlib 的 `V>0∧WH=0 → NumericalError` 零值约定、
  `reconstruction_err_` 在 KL 损失下为 sqrt(2KL) 而 NMFlib 恒为 Frobenius 范数、
  随机初始化分布（同尺度 `sqrt(mean/k)`、不同 RNG）。相同 seed 不意味着相同初始 W/H，
  结论基于多 seed 统计与独立目标，不做未对齐的 W/H 逐元素比较。

复现（需要 conda 与 moon 工具链）：

```sh
cd validation/sklearn_reference
pip install -r requirements.txt   # numpy/scipy/scikit-learn/matplotlib 固定版本
python generate_data.py && python selftest.py
python run_experiments.py         # 完整网格 -> results/results.csv + metadata.json
python analyze_results.py         # summary.csv / analysis.md / plots/
python run_experiments.py --smoke # CI 使用的轻量子集
```

MoonBit 侧通过 `moonbit_runner/`（独立模块，moon.work 本地工作区依赖本仓库）
只用公开 API 拟合并输出 W/H，供 Python 独立评估。

`src/fixtures_test.mbt`（21 项，由 `scripts/gen_fixtures.py` 生成）把实测筛选出的
sklearn 参考目标值/重构误差固化为字面量：同配置 fit 后断言完整目标与重构误差相对偏差
≤5%、W/H 非负有限、维度正确。**5% 容差经过实测验证**：仅纳入实测 seed-42 偏差 ≤3%
且参考量级足够（J≥1e-3、重构误差≥1e-2）的组合；全部实测偏差（含未纳入组合及原因）
记录于 `validation/sklearn_reference/results/fixtures_selection.json`。
近零目标（无正则精确恢复场景）的相对偏差无意义，不纳入 fixture 断言，其实测绝对值
与分析见 `results/analysis.md`。

## 与现有实现的关系

mooncakes.io 已有 `Juwan-Hwang/moon-certified/ml/nmf`（大仓库子包，仅 MU-Frobenius）。本库为专用 NMF 库：多求解器/多损失/NNDSVD 初始化/显式错误处理/数值验证。未复制其源码。

## 参考来源

Lee & Seung (1999/2001)；Hsieh & Dhillon (KDD 2011)；Boutsidis & Gallopoulos (2008)；Hoyer (JMLR 2004)；scikit-learn `decomposition.NMF`（API 参考，BSD，未复制代码）。

## 许可证

Apache-2.0（见 [LICENSE](LICENSE)）