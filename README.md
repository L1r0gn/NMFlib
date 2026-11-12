# NMFlib

纯 MoonBit 实现的非负矩阵分解（NMF）库。`V ≈ W·H`，W、H 非负。

> 状态：v0.1.0 开发中，当前里程碑 T01–T07（MU-Frobenius 完整闭环）。开发规格见 [PLAN.md](PLAN.md)，架构见 [docs/architecture.md](docs/architecture.md)。

<!-- 远程仓库确认后启用 CI 徽章：
[![CI](https://github.com/<owner>/nmf/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/nmf/actions/workflows/ci.yml)
-->

## 功能支持情况

| 功能 | 状态 |
|---|---|
| MU-Frobenius 求解器 | T01–T07 实现中 |
| Random 初始化（SplitMix64，seed 可复现） | T01–T07 实现中 |
| fit / fit_transform / transform / inverse_transform / reconstruction_error | T01–T07 实现中 |
| 完整错误处理（`Result` + 7 类错误枚举，不静默修改输入） | T01–T07 实现中 |
| CD-Frobenius 求解器 | 计划（T08） |
| NNDSVD / NNDSVDa 初始化 | 计划（T09–T10） |
| MU-KL 散度 | 计划（T11） |
| L1/L2 正则（进入优化目标） | 计划（T12） |
| Hoyer 稀疏度指标 | 计划（T13） |

纯 MoonBit、零第三方依赖（仅 core），支持 wasm / wasm-gc / js / native 四个后端。

## 安装

发布到 mooncakes.io 后：

```sh
moon add L1r0gn/nmf
```

当前可直接克隆源码使用（尚未发布）。

## 快速上手

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
```

未训练的模型调用 `w()` / `h()` / `transform()` 等接口会返回 `Err(NotFitted)`。完整可运行示例见 `examples/quickstart`（T16 交付；T07 完成后提供等价演示）。

## 运行测试

```sh
moon check --deny-warn   # 类型与警告检查
moon test                # 全部测试（默认 wasm 后端）
moon test --target js    # JS 后端
moon test --target wasm-gc
moon test --target native
moon fmt --check         # 格式检查
```

CI（`.github/workflows/ci.yml`）在 push / PR 时运行三个 Job：格式与 check、完整测试套件、三后端 check+test。工具链版本采用 fail-closed 固定策略（见 `scripts/ci-install-moonbit.sh`）。

## 与现有实现的关系

mooncakes.io 已有 `Juwan-Hwang/moon-certified/ml/nmf`（大仓库子包，仅 MU-Frobenius）。本库为专用 NMF 库：多求解器/多损失/NNDSVD 初始化/显式错误处理/数值验证。差异详见 PLAN.md 第 2 节。未复制其源码。

## 参考来源

Lee & Seung (1999/2001)；Hsieh & Dhillon (KDD 2011)；Boutsidis & Gallopoulos (2008)；Hoyer (JMLR 2004)；scikit-learn `decomposition.NMF`（API 参考，BSD，未复制代码）。

## 许可证

Apache-2.0（见 [LICENSE](LICENSE)）