"""sakura 翻译服务生命周期管理

- 启动: 子进程跑 sakura_server.py(pythonw, 无窗口), 日志重定向到 sakura\\gui_server.log
- 停止: 仅终止本管理器启动的进程; 外部启动的服务不强制杀
- 健康: QTimer 轮询 http://127.0.0.1:8080/health (线程内请求防卡 UI)
"""
import subprocess
import sys
import threading
import urllib.request

from PySide6.QtCore import QObject, QTimer, Signal

from paths import PYTHONW, SAKURA_SERVER, SAKURA_LOG, SAKURA_HEALTH

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class SakuraManager(QObject):
    status_changed = Signal(str)   # "running" | "loading" | "stopped"
    log = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None          # 本管理器启动的进程
        self._checking = False
        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.check)
        self.timer.start()
        self.check()

    # ---------- 健康检测 ----------

    def check(self):
        if self._checking:
            return
        self._checking = True
        threading.Thread(target=self._probe, daemon=True).start()

    def _probe(self):
        ok = False
        try:
            with urllib.request.urlopen(SAKURA_HEALTH, timeout=2) as r:
                ok = r.status == 200
        except Exception:
            ok = False
        if self.proc is not None and self.proc.poll() is not None:
            self.proc = None
        self.status_changed.emit("running" if ok else
                                 ("loading" if self.proc is not None else "stopped"))
        self._checking = False

    # ---------- 启停 ----------

    def start(self):
        if self.is_running():
            self.log.emit("服务已在运行")
            return
        try:
            import os
            os.makedirs(os.path.dirname(SAKURA_LOG), exist_ok=True)
            logf = open(SAKURA_LOG, "ab", buffering=0)
            self.proc = subprocess.Popen(
                [PYTHONW, SAKURA_SERVER],
                cwd=os.path.dirname(SAKURA_SERVER),
                creationflags=CREATE_NO_WINDOW,
                stdout=logf, stderr=logf)
            self.log.emit("正在启动翻译服务（模型加载约 20 秒）…")
            self.status_changed.emit("loading")
        except Exception as e:
            self.proc = None
            self.log.emit(f"启动失败: {e}")
            self.status_changed.emit("stopped")

    def stop(self):
        p = self.proc
        if p is None or p.poll() is not None:
            if not self.is_running():
                self.log.emit("服务未运行，无需停止")
                return
            self.log.emit("服务由外部启动，本界面不强制停止（可用 start_sakura.bat 管理）")
            return
        self.log.emit("正在停止翻译服务…")
        try:
            p.terminate()
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=3)
        except Exception as e:
            self.log.emit(f"停止失败: {e}")
        self.proc = None
        # 本机异常: venv 跑 sakura_server 时会派生系统 python 子进程占用 8080，
        # 直接终止父进程无法释放端口，需清理仍在监听 8080 的 sakura 进程
        if self.is_running():
            killed = self._kill_8080_listener()
            if killed:
                self.log.emit(f"已清理残留监听进程: {killed}")
        self.check()

    def _kill_8080_listener(self) -> list:
        """终止仍监听 127.0.0.1:8080 且命令行含 sakura_server 的进程"""
        import os
        killed = []
        try:
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True,
                errors="replace", creationflags=CREATE_NO_WINDOW).stdout
            pids = set()
            for line in out.splitlines():
                if "127.0.0.1:8080" in line and "LISTENING" in line.upper():
                    pids.add(line.split()[-1])
            for pid in pids:
                if pid.isdigit() and int(pid) == os.getpid():
                    continue
                if self._is_sakura_pid(pid):
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True, creationflags=CREATE_NO_WINDOW)
                    killed.append(pid)
        except Exception:
            pass
        return killed

    @staticmethod
    def _is_sakura_pid(pid: str) -> bool:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                capture_output=True, text=True, errors="replace",
                creationflags=CREATE_NO_WINDOW, timeout=8).stdout
            return "sakura_server" in out
        except Exception:
            return False

    # ---------- 查询 ----------

    def is_running(self) -> bool:
        try:
            with urllib.request.urlopen(SAKURA_HEALTH, timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def is_owned(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
