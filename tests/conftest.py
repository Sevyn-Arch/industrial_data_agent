"""pytest 公共夹具与路径引导。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_core import IndustrialDataAgent  # noqa: E402


def build_frame(rows: int = 300, lag: int = 3) -> pd.DataFrame:
    """构造带有「已知时滞 + 已知异常点 + 缺失值」的合成工业数据集。"""
    rng = np.random.default_rng(42)
    index = np.arange(rows)
    inlet = pd.Series(245 + 6 * np.sin(index / 20.0) + rng.normal(0, 0.8, rows))
    inlet.iloc[50] += 40.0  # 已知异常点：第 50 行
    steam = pd.Series(12.5 + 0.3 * np.sin(index / 15.0) + rng.normal(0, 0.2, rows))
    outlet = 0.9 * inlet.shift(lag).bfill() + 0.5 * steam + rng.normal(0, 0.9, rows)
    # 目标变量保持较强的可解释信号（便于模型类断言稳定）：yield = 常数 + 系数×过程变量 + 噪声
    yield_pct = 55 + 0.35 * outlet + 1.5 * steam + rng.normal(0, 0.4, rows)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="5min"),
            "inlet_temp_c": inlet.round(3),
            "steam_flow_t_h": steam.round(3),
            "outlet_temp_c": outlet.round(3),
            "yield_pct": yield_pct.round(3),
            "grade": rng.choice(["A", "B", "C"], size=rows),
        }
    )
    frame.loc[[10, 11], "steam_flow_t_h"] = np.nan
    return frame


@pytest.fixture(scope="session")
def sample_frame() -> pd.DataFrame:
    """会话级合成数据（含 3 步时滞与 1 个异常点）。"""
    return build_frame()


@pytest.fixture()
def sample_csv(tmp_path: Path, sample_frame: pd.DataFrame) -> Path:
    """把合成数据写入临时目录的 CSV，返回文件路径。"""
    target = tmp_path / "process_data.csv"
    sample_frame.to_csv(target, index=False, encoding="utf-8")
    return target


@pytest.fixture()
def agent(tmp_path: Path, sample_frame: pd.DataFrame) -> IndustrialDataAgent:
    """离线（模板模式）Agent 实例：数据目录与输出目录均为临时目录。"""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "charts").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    sample_frame.to_csv(data_dir / "process_data.csv", index=False, encoding="utf-8")
    instance = IndustrialDataAgent(data_dir=data_dir, output_dir=output_dir, use_llm=False)
    return instance


@pytest.fixture(scope="session")
def project_data_file(sample_frame: pd.DataFrame):
    """在项目 data 目录下临时落一份数据（沙箱只允许读取项目 data/output 目录）。

    用于需要真实沙箱执行、且必须通过 ``load_data()`` 读取项目数据目录的测试；
    测试结束后自动删除该文件。
    """
    from analytics import DATA_DIR

    target = DATA_DIR / "pytest_shared_data.csv"
    sample_frame.to_csv(target, index=False, encoding="utf-8")
    try:
        yield target
    finally:
        target.unlink(missing_ok=True)
