"""
Official Qlib Alpha158 Feature Adapter.
Directly wraps and reuses microsoft/qlib's official Alpha158 data handler and loader specifications.
"""
from typing import Dict, Any, List, Tuple, Optional
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.features.qlib_alpha158")

class OfficialQlibAlpha158:
    """
    官方 Qlib Alpha158 适配器
    直接调用 microsoft/qlib 的 qlib.contrib.data.handler.Alpha158 与 Alpha158DL 配置
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
        实例化 Qlib 官方 Alpha158 Data Handler
        """
        logger.info(f"Instantiating official qlib.contrib.data.handler.Alpha158 for {instruments}...")
        handler_kwargs = {**self.extra_kwargs, **kwargs}
        if fit_start_time is not None:
            handler_kwargs["fit_start_time"] = fit_start_time
        if fit_end_time is not None:
            handler_kwargs["fit_end_time"] = fit_end_time
            
        handler = self.handler_cls(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            **handler_kwargs
        )
        return handler

# Alias for backward compatibility
QlibAlpha158Features = OfficialQlibAlpha158

