import os
import io
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
from typing import Any, Dict, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from content_utils import extract_obj, extract_sol


class GenericRuntime:
    GLOBAL_DICT = {}  # Global variables available in the runtime
    LOCAL_DICT = None  # Local variables available in the runtime, can be None if not used
    HEADERS = []  # List of code snippets to execute at initialization

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r"(\s|^)?input\(", code_piece) or regex.search(r"(\s|^)?os.system\(", code_piece):
            raise RuntimeError("Input or system commands are not allowed")
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars.get("answer")

class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        "datetime": datetime.datetime,
        "timedelta": dateutil.relativedelta.relativedelta,
        "relativedelta": dateutil.relativedelta.relativedelta,
        }

class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()

class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {"dict": CustomDict}

class PythonExecutor:
    def __init__(self, runtime: Optional[Any] = None, timeout_length: int = 5) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.timeout_length = timeout_length

    def process_generation_to_code(self, gens: list) -> list:
        return [g.split("\n") if g is not None else None for g in gens]


    @staticmethod
    def execute(code, runtime=None, timeout_length=150):
        """Run one code snippet. timeout_length kept for API compat; wall timeout is at batch_apply.

        Do NOT use signal-based timeout_decorator here: RewardLoopWorker is multithreaded,
        and SIGALRM only works in the main thread. Solver hangs are bounded by Gurobi TimeLimit.

        Do NOT use redirect_stdout: it mutates process-global sys.stdout and races under threads.
        Capture via a per-runtime print() injected into exec globals instead.
        """
        import builtins
        try:
            if runtime is None:
                runtime = GenericRuntime()
            if not code:
                return "", "Empty code"
            program_io = io.StringIO()

            def _capture_print(*args, **kwargs):
                kwargs.pop("file", None)
                builtins.print(*args, file=program_io, **kwargs)

            runtime.inject({"print": _capture_print})
            runtime.exec_code("\n".join(code))
            program_io.seek(0)
            result = program_io.read()
            if result == "":
                runtime.exec_code("\n".join(code[:-1]))
                result = runtime.eval_code(code[-1])
            report = "Done"
            str(result)
            if result is not None:
                pickle.dumps(result)  # Serialization check
        except Exception:
            result = ""
            report = traceback.format_exc().split("\n")[-2]
        return result, report

    def apply(self, code):
        return self.batch_apply([code])[0]

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, batch_code):
        """Execute codes in a thread pool (no fork).

        Replaces pebble ProcessPool: forking from Ray's multithreaded RewardLoopWorker
        caused zombie children and permanent pool.join() hangs.
        """
        all_code_snippets = self.process_generation_to_code(batch_code)
        n = len(all_code_snippets)
        all_exec_results = [("", "Timeout Error")] * n

        # Cover Gurobi TimeLimit=60 + Python overhead; ignore tiny default timeout_length=5
        wall_timeout = max(int(self.timeout_length), 75)
        max_workers = max(1, min(n, max(1, (os.cpu_count() or 4) // 4)))

        def _run_one(idx_code):
            idx, code = idx_code
            # Fresh runtime per task — shared GenericRuntime would race across threads
            runtime = type(self.runtime)()
            return idx, self.execute(code, runtime=runtime, timeout_length=wall_timeout)

        progress_bar = tqdm(total=n, desc="Execute") if n > 100 else None
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            future_to_idx = {
                pool.submit(_run_one, (i, code)): i
                for i, code in enumerate(all_code_snippets)
            }
            try:
                for fut in as_completed(future_to_idx, timeout=wall_timeout * max(1, (n + max_workers - 1) // max_workers)):
                    idx = future_to_idx[fut]
                    try:
                        _, result = fut.result(timeout=0)
                        all_exec_results[idx] = result
                    except Exception as error:
                        all_exec_results[idx] = ("", f"ExecutorError: {error}")
                    if progress_bar is not None:
                        progress_bar.update(1)
            except FuturesTimeoutError:
                for fut, idx in future_to_idx.items():
                    if not fut.done():
                        all_exec_results[idx] = ("", "Timeout Error")
                        if progress_bar is not None:
                            progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()
            # wait=False: never block reward on a stuck worker thread
            pool.shutdown(wait=False, cancel_futures=True)

        batch_obj = []
        batch_sol = []
        batch_report = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            res, report = str(res).strip(), str(report).strip()
            batch_obj.append(extract_obj(res))
            sol = extract_sol(res)
            batch_sol.append(sol)
            batch_report.append(report)
        # 注意：不要在这里 print 大批量结果——RewardLoopWorker 的 stdout 走 Ray 管道，
        # 巨量输出会写满管道导致 write() 永久阻塞（2026-07-13 5h 死锁事故根因）。
        return batch_obj, batch_sol, batch_report

