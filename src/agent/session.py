from datetime import datetime


class SessionState:
    """管理 agent 的会话状态和配置"""

    def __init__(self, thread_id: str = "default_session"):
        self.thread_id = thread_id

    @property
    def config(self):
        """获取当前 LangGraph 运行配置"""
        return {"configurable": {"thread_id": self.thread_id}}

    def new_session(self):
        """开始全新会话（清除记忆）"""
        self.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"🔄 已开启新会话: {self.thread_id}")
        return self.config


# 全局唯一的会话状态实例
session = SessionState()
