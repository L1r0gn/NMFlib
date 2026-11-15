# NMFlib

纯 MoonBit 实现的非负矩阵分解（NMF）库。`V ≈ W·H`，W、H 非负。

> 状态：v0.1.0 开发中。**T01–T09 已完成**：MU-Frobenius、CD-Frobenius 与 NNDSVD 初始化可用，
> 107 项测试在 wasm / wasm-gc / js / native 四后端全部通过。

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
| NNDSVDa 初始化（零元素均值填充） | 计划（T10） |
| MU-KL 散度 | 计划（T11） |
| L1/L2 正则（进入优化目标） | 计划（T12） |
| Hoyer 稀疏度指标 | 计划（T13） |

选择尚未实现的损失/初始化（如 NNDSVDa）、不支持的 CD+KL 组合或非零正则时，`fit` 返回 `Err(InvalidConfig)`（注明目标任务号），不会静默替换。

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

### NMFConfig 使用方法

全部参数为标签参数，除 `n_components` 外均有默认值：

```moonbit
let config = @nmf.NMFConfig::new(
  n_components=2,            // 分解秩 k，必填，1 ≤ k ≤ min(rows, cols)
  solver=@nmf.Solver::MU,    // 默认 MU；可改为 CD（循环坐标下降，仅支持 Frobenius）
  loss=@nmf.Loss::Frobenius, // 默认 Frobenius（KL 计划 T11）
  init=@nmf.Init::Random,    // 默认 Random；可改为 NNDSVD（确定性；NNDSVDa 计划 T10）
  max_iter=500,              // 默认 200；耗尽未达标时 converged=false
  tol=1.0e-6,                // 默认 1e-4；相邻目标相对变化 |ΔF|/max(1,|F|) < tol 即收敛
  seed=2026UL,               // 默认 42；同 seed 结果逐位一致
)
```

训练结果通过 `model.w()` / `model.h()`（防御性拷贝）、`model.n_iter()`、
`model.loss_value()`（真实目标 ½‖V−WH‖²_F）、`model.converged()` 读取；
`model.reconstruction_error(v)` 返回范数 ‖V−WH‖_F = sqrt(2·loss)。
未训练的模型调用以上接口一律返回 `Err(NotFitted)`；fit 失败会回到未训练状态。
`model.transform(x)` 固定 H 对新数据解 W；`model.inverse_transform(w)` 计算 W·H。
`solver=@nmf.Solver::CD` 时 fit 与 transform 都使用 CD 坐标下降，不会回退到 MU。
`init=@nmf.Init::NNDSVD` 使用内部幂迭代截断 SVD；初始化确定性、不依赖 `seed`，只影响
fit 的初始 W/H（transform 仍沿用既有行为，用 Random 初始化新数据的 W）。内部 SVD 不是
公共 API；标准 NNDSVD 不填充零元素（NNDSVDa 的均值填充属于 T10）。

## 运行测试

```sh
moon check --deny-warn   # 类型与警告检查
moon test                # 全部 107 项测试（默认 wasm 后端）
moon test --target js    # JS 后端（真实执行全部测试）
moon test --target wasm-gc
moon test --target native
moon fmt --check         # 格式检查
```

测试分层：Matrix 单元/白盒内核 → Error 行为 → SplitMix64/Random → Frobenius 损失 →
MU 求解器 → CD 求解器（精确坐标步、目标不增、零曲率/零行列/极小值边界）→
内部 SVD（幂迭代/deflation/秩不足/重复奇异值/极小值）→ NNDSVD（正负部选对、零退化、
非负、确定性）→ NMF API → 端到端验收（普通矩阵、已知低秩、全零、非法输入、seed 复现、
未收敛状态）+ 200 组随机属性扫描。CD 与 NNDSVD 均有独立白盒测试；NNDSVD 的初始目标值
在固定低秩夹具上与 Random 平均值实测比较。

CI（`.github/workflows/ci.yml`）在 push / PR 时运行三个 Job：格式与 check、完整测试套件、
三后端 check+test（后者的 matrix 展开为 native / js / wasm-gc 三个运行实例，共 5 个 Job 实例）。
工具链版本采用 fail-closed 固定策略（见 `scripts/ci-install-moonbit.sh`）：官方安装脚本安装后
断言版本号，不符即失败。

CI 已在 GitHub Actions 实际运行并通过（2026-11-15，5 个 Job 实例全部成功）。
上述命令亦已在本地按 CI 相同配置验证通过（本地工具链 0.1.20261028；CI 锁定 0.1.20261104）。

## 与现有实现的关系

mooncakes.io 已有 `Juwan-Hwang/moon-certified/ml/nmf`（大仓库子包，仅 MU-Frobenius）。本库为专用 NMF 库：多求解器/多损失/NNDSVD 初始化/显式错误处理/数值验证。未复制其源码。

## 参考来源

Lee & Seung (1999/2001)；Hsieh & Dhillon (KDD 2011)；Boutsidis & Gallopoulos (2008)；Hoyer (JMLR 2004)；scikit-learn `decomposition.NMF`（API 参考，BSD，未复制代码）。

## 许可证

Apache-2.0（见 [LICENSE](LICENSE)）