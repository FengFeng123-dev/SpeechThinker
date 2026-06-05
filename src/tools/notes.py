from datetime import datetime
from langchain_core.tools import tool
from config.settings import settings


@tool
def save_note(content: str, title: str = "未命名笔记") -> str:
    """将用户的笔记内容保存到本地文件中。当用户说"帮我记一下"、"保存这个"时调用此工具。

    Args:
        content: 要保存的笔记正文内容
        title: 笔记标题，默认为"未命名笔记"
    """
    note_dir = settings.NOTES_DIR
    note_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = note_dir / filename
    filepath.write_text(
        f"标题: {title}\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"{content}",
        encoding="utf-8"
    )
    return f"✅ 笔记已保存至：{filepath}"
