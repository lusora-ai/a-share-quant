"""
Official Qlib Alpha158 Feature Adapter.
Directly wraps and reuses microsoft/qlib's official Alpha158 data handler and loader specifications.

Project 5D executable label (mandatory, never the Qlib default label):
    signal at t close -> entry = open[t+1] -> exit = close[t+5]
    Qlib expression: Ref($close, -5) / Ref($open, -1) - 1
    Label is cross-sectionally normalized with qlib CSZScoreNorm (NOT a percentile rank).
"""
from typing import Dict, Any, List, Tuple, Optional
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.features.qlib_alpha158")

# ---- Project 5D executable label (single source of truth) ----
EXEC_LABEL_5D_EXPRESSION = "Ref($close, -5) / Ref($open, -1) - 1"
EXEC_LABEL_5D_NAME = "exec_return_5d_cszscore"
EXEC_LABEL_5D_HORIZON = 5

# Learn processors applied to the label: drop immature/NaN labels, then cross-sectional z-score.
EXEC_LABEL_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]


def build_exec_label_5d_spec() -> Dict[str, Any]:
    """
    返回项目 5D executable label 的完整 spec，用于实验 metadata 记录
    """
    return {
        "name": EXEC_LABEL_5D_NAME,
        "expression": EXEC_LABEL_5D_EXPRESSION,
        "horizon": EXEC_LABEL_5D_HORIZON,
        "signal_time": "t close (signal generated after close on day t)",
        "entry_time": "t+1 open (entry = open[t+1])",
        "exit_time": "t+5 close (exit = close[t+5])",
        "normalization": "qlib CSZScoreNorm on label (cross-sectional z-score; NOT a percentile rank)",
        "learn_processors": ["DropnaLabel", "CSZScoreNorm(label)"],
    }


class OfficialQlibAlpha158:
    """
    官方 Qlib Alpha158 适配器
    直接调用 microsoft/qlib 的 qlib.contrib.data.handler.Alpha158 与 Alpha158DL 配置
    显式注入项目 5D executable label，绝不依赖 Alpha158 默认 label
    """
    def __init__(self, **kwargs):
        self.handler_cls = Alpha158
        self.loader_cls = Alpha158DL
        self.extra_kwargs = kwargs

    @classmethod
    def get_feature_config(cls) -> Tuple[List[str], List[str]]:
        """
        直接获取 Qlib 官方 Alpha158 的 158 个特征计算表达式与字段名
        """
        fields, names = Alpha158DL.get_feature_config()
        return fields, names

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """
        获取官方 158 个 Alpha 特征列名称列表
        """
        _, names = cls.get_feature_config()
        return names

    def build_handler_kwargs(
        self,
        instruments: str = "csi300",
        start_time: str = "2018-01-01",
        end_time: Optional[str] = None,
        fit_start_time: Optional[str] = None,
        fit_end_time: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建传给官方 Alpha158 handler 的完整 kwargs。
        label 永远是项目 5D executable label（显式传入，禁止使用 Qlib 默认 label）。
        """
        handler_kwargs: Dict[str, Any] = {
            "instruments": instruments,
            "start_time": start_time,
            "end_time": end_time,
            # 显式项目 label: ([expression], [name])
            "label": ([EXEC_LABEL_5D_EXPRESSION], [EXEC_LABEL_5D_NAME]),
            "learn_processors": [dict(p) for p in EXEC_LABEL_LEARN_PROCESSORS],
        }
        if fit_start_time is not None:
            handler_kwargs["fit_start_time"] = fit_start_time
        if fit_end_time is not None:
            handler_kwargs["fit_end_time"] = fit_end_time
        handler_kwargs.update(self.extra_kwargs)
        handler_kwargs.update(kwargs)
        return handler_kwargs

    def create_handler_instance(
        self,
        instruments: str = "csi300",
        start_time: str = "2018-01-01",
        end_time: Optional[str] = None,
        fit_start_time: Optional[str] = None,
        fit_end_time: Optional[str] = None,
        **kwargs
    ):
        """
        实例化 Qlib 官方 Alpha158 Data Handler（带项目 5D executable label）
        """
        logger.info(
            f"Instantiating official qlib.contrib.data.handler.Alpha158 for {instruments} "
            f"with project 5D executable label '{EXEC_LABEL_5D_NAME}' ({EXEC_LABEL_5D_EXPRESSION})..."
        )
        handler_kwargs = self.build_handler_kwargs(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            **kwargs
        )
        handler = self.handler_cls(**handler_kwargs)
        return handler

# Alias for backward compatibility
QlibAlpha158Features = OfficialQlibAlpha158
