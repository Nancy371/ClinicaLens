"""
Hospital Agent SDK 本地替代包。

提供 BaseDoctorAgent 基类和 Actions 接口，
使项目可独立运行，不依赖 PyPI 上不存在的 hospital-agent-sdk。
"""

from .base import BaseDoctorAgent, Actions

__all__ = ["BaseDoctorAgent", "Actions"]