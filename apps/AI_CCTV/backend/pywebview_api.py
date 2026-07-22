from pathlib import Path

import webview

from .state import state


class Api:
    """pywebview js_api로 노출되는 네이티브 파일 다이얼로그.
    프론트엔드에서 window.pywebview.api.<method>(...)로 호출한다.
    """

    def open_video_dialog(self):
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=("Video Files (*.mp4;*.avi;*.mkv)", "All files (*.*)"),
        )
        if not result:
            return []
        for p in result:
            state.add_video(Path(p))
        return [str(p) for p in result]

    def save_excel_dialog(self):
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename="CCTV_조사_집계표.xlsx",
            file_types=("Excel Files (*.xlsx)",),
        )
        if not result:
            return None
        return str(result if isinstance(result, str) else result[0])
