# 升级报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 仓库名 | scikit-optimize |
| 升级时间 | 2026-03-14 |
| 升级状态 | ✅ 成功 |

## Python 版本

| 升级前 | 升级后 |
|--------|--------|
| >=3.6 | >=3.13 |

## 依赖变更

| 依赖 | 升级前 | 升级后 |
|------|--------|--------|
| numpy | >=1.13.3 | >=2.4.3 |
| scipy | >=0.19.1 | >=1.17.1 |
| scikit-learn | >=0.20.0 | >=1.8.0 |
| matplotlib | >=2.0.0 | >=3.10.8 |
| pytest | (无版本要求) | >=9.0.2 |
| pyaml | >=16.9 | >=26.2.1 |
| joblib | >=0.11 | >=1.5.3 |

## 代码修改

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| setup.py | distutils 迁移 | 移除 distutils 兼容代码，直接使用 setuptools |
| setup.py | Python 2 兼容代码移除 | 移除 __builtin__ 兼容代码 |
| setup.py | Python 版本声明 | 更新 classifiers 和 python_requires |
| conftest.py | distutils.version 迁移 | 改用 packaging.version.Version |
| skopt/space/transformers.py | NumPy 类型别名 | np.int → int |
| skopt/learning/forest.py | sklearn API 更新 | criterion='mse' → 'squared_error' |
| skopt/learning/forest.py | sklearn API 更新 | max_features='auto' → 1.0 |
| skopt/learning/gbrt.py | sklearn tags 系统 | 添加 __sklearn_tags__ 方法 |
| skopt/learning/gbrt.py | NumPy API 更新 | np.in1d → np.isin |
| skopt/optimizer/optimizer.py | is_regressor 兼容 | 添加异常处理避免对非对象调用 |
| skopt/utils.py | is_regressor 兼容 | 添加异常处理避免对非对象调用 |
| skopt/tests/test_common.py | pytest API 更新 | pytest.warns(None) → warnings.catch_warnings |

## 测试结果

| 测试类型 | 结果 |
|----------|------|
| 通过 | 439 passed |
| 失败 | 0 failed |
| 警告 | 153 warnings |

| 升级前 | 升级后 |
|--------|--------|
| N/A | ✅ 439 passed, 0 failed |

## 主要兼容性问题及解决方案

### 1. distutils 模块移除（Python 3.12+）
- **问题**: `from distutils.version import LooseVersion` 失败
- **解决**: 改用 `from packaging.version import Version`

### 2. NumPy 2.0 类型别名移除
- **问题**: `np.int` 不再可用
- **解决**: 改用 Python 内置 `int` 类型

### 3. scikit-learn API 变更
- **问题**: `criterion='mse'` 已弃用
- **解决**: 改用 `criterion='squared_error'`
- **问题**: `max_features='auto'` 已移除
- **解决**: 改用 `max_features=1.0`（等价于使用所有特征）

### 4. scikit-learn tags 系统变更
- **问题**: `is_regressor()` 依赖新的 tags 系统，`GradientBoostingQuantileRegressor` 未正确标识为回归器
- **解决**: 添加 `__sklearn_tags__()` 方法显式声明 `estimator_type = "regressor"`

### 5. NumPy API 更新
- **问题**: `np.in1d` 已弃用
- **解决**: 改用 `np.isin`

### 6. is_regressor 对非对象调用
- **问题**: 新版 sklearn 的 `is_regressor()` 对非对象（如 int、None）调用会抛出 AttributeError
- **解决**: 在调用前添加 try-except 捕获异常

### 7. pytest.warns API 变更
- **问题**: `pytest.warns(None)` 在新版 pytest 中不再支持
- **解决**: 改用标准库 `warnings.catch_warnings(record=True)`

## 备注

所有测试均已通过，代码已成功升级到 Python 3.13 + 最新依赖版本。主要修改集中在：
1. 移除 Python 2 兼容代码
2. 适配 NumPy 2.0 API 变更
3. 适配 scikit-learn 1.8.0 API 变更
4. 修复测试代码以适配新版 pytest
