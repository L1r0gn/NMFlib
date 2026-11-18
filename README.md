# NMFlib

**NMFlib** 是一个基于 MoonBit 实现的非负矩阵分解（Non-negative Matrix Factorization，NMF）数值计算库。

通过将非负矩阵分解为两个低秩非负矩阵
$
V \approx WH
$
为数据降维、潜在特征提取和数据分析提供基础算法支持。

[![CI](https://github.com/L1r0gn/NMFlib/actions/workflows/ci.yml/badge.svg)](https://github.com/L1r0gn/NMFlib/actions/workflows/ci.yml)

- GitHub：[L1r0gn/NMFlib](https://github.com/L1r0gn/NMFlib)
- 版本：`0.1.0`（Preview）
- 许可证：Apache-2.0

## 核心功能

**求解算法**
- MU：乘法更新算法。
- CD：循环坐标下降算法。

**损失函数**
- Frobenius 范数。
- Generalized KL Divergence（广义 KL 散度）。

**初始化方法**
- Random：基于 SplitMix64，固定 seed 可复现。
- NNDSVD：基于截断 SVD 的非负双奇异值初始化。
- NNDSVDa：使用输入矩阵均值填充 NNDSVD 结果中的精确零元素。

**正则化与评价指标**
- L1、L2 及组合正则化，作用于 W 和 H。
- Hoyer Sparsity：衡量非负向量的稀疏程度。

**模型 API**
- `fit` / `fit_transform`：模型训练。
- `transform`：固定训练得到的 H，对新数据求解 W。
- `inverse_transform`：根据 W、H 重构矩阵。
- `reconstruction_error`：计算 Frobenius 重构误差。
- `loss_value`：获取包含正则化项的完整优化目标。
- `sparsity`：计算 Hoyer 稀疏度。

提供矩阵校验、统一错误处理、收敛判断、模型状态保护和随机种子可复现能力。

## 安装

发布到 Mooncakes 后，可以使用：

```bash
moon add L1r0gn/nmf
```

当前也可以通过源码运行：

```bash
git clone https://github.com/L1r0gn/NMFlib.git
cd NMFlib
moon run examples/quickstart
```

## 快速开始

以下示例展示矩阵构造、模型配置和训练：

```moonbit
fn main {
  let v = match @nmf.Matrix::new(
    3,
    3,
    [1.0, 2.0, 0.0,
     3.0, 0.0, 4.0,
     0.0, 5.0, 6.0],
  ) {
    Ok(m) => m
    Err(e) => abort("Matrix creation failed: \{e}")
  }

  let config = @nmf.NMFConfig::new(
    n_components=2,
    seed=42UL,
  )

  let model = @nmf.NMF::new(config)

  match model.fit(v) {
    Ok(_) => ()
    Err(e) => abort("NMF failed: \{e}")
  }

  let w = match model.w() {
    Ok(m) => m
    Err(e) => abort("Failed to get W: \{e}")
  }

  let h = match model.h() {
    Ok(m) => m
    Err(e) => abort("Failed to get H: \{e}")
  }

  let error = match model.reconstruction_error(v) {
    Ok(x) => x
    Err(e) => abort("Failed to calculate error: \{e}")
  }

  println("W: \{w.rows()}x\{w.cols()}")
  println("H: \{h.rows()}x\{h.cols()}")
  println("Reconstruction error: \{error}")
}
```

运行：

```bash
moon run examples/quickstart
```

实际输出：

```text
W: 3x2
H: 2x3
iterations=64, converged=true
reconstruction error = 1.8065620213588327
```

## 模型配置

所有参数均为标签参数，只有 `n_components` 必填。

```moonbit
let config = @nmf.NMFConfig::new(
  n_components=3,
  solver=@nmf.Solver::MU,
  loss=@nmf.Loss::KLDivergence,
  init=@nmf.Init::Random,
  max_iter=500,
  tol=1.0e-4,
  seed=42UL,
  l1_reg=0.001,
  l2_reg=0.0,
)
```

正则化目标：

$$
\Omega(W,H)=
\lambda_1(\lVert W\rVert_1+\lVert H\rVert_1)
+\frac{\lambda_2}{2}
(\lVert W\rVert_F^2+\lVert H\rVert_F^2)
$$

`loss_value()` 返回包含正则项的完整优化目标；`reconstruction_error()` 始终返回不含正则项的 Frobenius 重构误差。

默认正则化参数均为 `0.0`。负数、NaN 和 Infinity 会被拒绝。

模型训练后，可以通过 `model.w()`、`model.h()`、`model.n_iter()` 和 `model.converged()` 获取结果。未训练的模型调用结果读取接口会返回 `NotFitted`。

## Hoyer 稀疏度

```moonbit
let result = @nmf.sparsity([1.0, 1.0, 0.0, 0.0])
```

计算公式：

$$
S(x)=
\frac{\sqrt{n}-\frac{\lVert x\rVert_1}{\lVert x\rVert_2}}
{\sqrt{n}-1}
$$

取值范围为 `[0, 1]`，数值越大表示向量越稀疏。该接口接受单个非负向量，不直接计算矩阵级平均稀疏度。

在 NMF 中：
- W 的列表示基向量；
- H 的行表示成分的样本激活。

可分别提取这些向量，再调用 `sparsity` 进行分析。

## 应用示例：Topic Modeling

项目提供完整的主题建模 MVP：

```bash
moon run examples/topic_modeling
```

示例使用固定文档语料构造文档—词语矩阵，通过 MU + 广义 KL + L1 完成主题分解，展示：

- 主题 Top-K 关键词提取；
- 文档主题激活分析；
- Hoyer 稀疏度计算；
- 新文档 `transform` 推理；
- OOV 词统计与处理。

详见 [examples/topic_modeling/README.md](examples/topic_modeling/README.md)。

## 数值验证与测试

项目使用 scikit-learn 作为独立参考实现，对三种有效求解组合进行数值对照，并通过独立目标函数验证结果。

参考验证工具位于 [validation/sklearn_reference](validation/sklearn_reference)。

当前测试体系包括：

- 209 项自动化测试；
- wasm、wasm-gc、native、js 四个后端；
- 21 个固定数值参考 Fixtures；
- Python Reference Smoke；
- Quickstart 与 Topic Modeling 自动运行检查。

运行测试：

```bash
moon fmt --check
moon check --deny-warn

moon test
moon test --target wasm-gc
moon test --target native
moon test --target js
```

GitHub Actions 自动执行代码检查、测试、后端兼容性和 Python 参考验证，并运行两个应用示例。

## 当前限制

- CD 求解器目前仅支持 Frobenius 损失。
- KL 模式下，若初始因子导致正输入位置出现零重构值，可能返回 `NumericalError`。
- NNDSVD/NNDSVDa 在乘法更新中可能遇到结构零锁定问题。
- 大规模矩阵性能与长期运行稳定性尚未进行系统评估。
- Topic Modeling 示例使用小型固定语料，主要用于展示算法流程，不代表完整 NLP 系统。

## 参考文献

- Lee & Seung：非负矩阵分解与乘法更新算法。
- Hsieh & Dhillon：非负矩阵分解的循环坐标下降算法。
- Boutsidis & Gallopoulos：NNDSVD 初始化方法。
- Hoyer：非负矩阵分解与稀疏性度量。
- scikit-learn：NMF API 与独立数值参考。

## 许可证

[Apache License 2.0](LICENSE)