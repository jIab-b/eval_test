import ast
import base64
import dataclasses
import multiprocessing
import re
import time
import os
import sys
import math
from pathlib import Path
from typing import Optional, Callable

import torch.cuda

from .utils import set_seed


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, "w")
        os.set_inheritable(fd, False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.file, flush=True)

    def log(self, key, value):
        self.print(f"{key}: {value}")


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


@dataclasses.dataclass
class BenchmarkResult:
    stats: Optional[Stats]
    error: Optional[str]


def _combine(a: int, b: int) -> int:
    return int(a + (a + b) * (a + b + 1) // 2)


def get_test_cases(file_name: str, seed: Optional[int]) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as E:
        print(f"Could not open test file`{file_name}`: {E}", file=sys.stderr)
        exit(113)

    tests = []
    lines = content.splitlines()
    for line in lines:
        parts = line.split(";")
        case = {}
        for part in parts:
            if not part.strip():
                continue
            key, sep, raw_val = part.partition(":")
            if not sep:
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                exit(113)
            key = key.strip()
            raw_val = raw_val.strip()
            if not key or not raw_val or not re.fullmatch(r"[a-zA-Z_]+", key):
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                exit(113)

            if raw_val[0] in "[(" and raw_val[-1] in "])":
                try:
                    val = ast.literal_eval(raw_val)
                except (SyntaxError, ValueError):
                    print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                    exit(113)
            else:
                try:
                    val = int(raw_val)
                except ValueError:
                    if not re.fullmatch(r"[a-zA-Z_]+", raw_val):
                        print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                        exit(113)
                    val = raw_val
            case[key] = val
        tests.append(TestCase(spec=line, args=case))

    if seed is not None:
        for test in tests:
            if "seed" in test.args:
                test.args["seed"] = _combine(test.args["seed"], seed)

    return tests


def calculate_stats(durations: list[float]) -> Stats:
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)
    avg = total / runs
    variance = sum(map(lambda x: (x - avg) ** 2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)
    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst))


def clone_data(data):
    if isinstance(data, tuple):
        return tuple(clone_data(x) for x in data)
    elif isinstance(data, list):
        return [clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data


class EvalRunner:
    """Base class for evaluation runners. Override methods as needed."""

    use_cutlass = False  # Set True to catch OpError
    use_batched_benchmark = True  # Use batched iterations for more stable timing
    batch_size = 15  # Number of iterations per benchmark batch
    use_large_cache_clear = True  # Use clear_l2_cache_large for B200

    def __init__(self):
        self._custom_kernel = None
        self._generate_input = None
        self._check_implementation = None
        self._compile_kernel = None
        self._clear_cache = None

    def setup(self):
        """Import task-specific modules. Called in subprocess."""
        if self.use_large_cache_clear:
            from .utils import clear_l2_cache_large as clear_cache
        else:
            from .utils import clear_l2_cache as clear_cache
        self._clear_cache = clear_cache

    def get_custom_kernel(self):
        raise NotImplementedError

    def get_generate_input(self):
        raise NotImplementedError

    def get_check_implementation(self):
        raise NotImplementedError

    def get_compile_kernel(self):
        """Return compile_kernel function or None if not needed."""
        return None

    def init_cuda(self):
        """Initialize CUDA context. Override for cutlass."""
        torch.cuda.init()

    def handle_kernel_error(self, e: Exception) -> tuple[bool, str]:
        """Handle exceptions from kernel execution."""
        print(f"Encountered {e}", file=sys.stderr)
        return False, str(e)

    def call_compile_kernel(self):
        """Call compile_kernel with appropriate arguments."""
        import inspect
        compile_kernel = self.get_compile_kernel()
        if compile_kernel is None:
            return
        sig = inspect.signature(compile_kernel)
        params = sig.parameters
        if not params:
            compile_kernel()
        else:
            kwargs = {}
            for name, param in params.items():
                if name == 'use_loop':
                    kwargs['use_loop'] = True
                elif param.default is inspect.Parameter.empty:
                    return  # Required param we can't provide
            compile_kernel(**kwargs)

    def compile_kernel_once(self) -> tuple[bool, Optional[str]]:
        """Compile kernel once before benchmarking."""
        try:
            self.init_cuda()
            self.setup()
            self.call_compile_kernel()
            torch.cuda.synchronize()
            return True, None
        except Exception as E:
            return False, f"Compilation failed: {E}"

    def run_single_test(self, test: TestCase) -> tuple[bool, str]:
        """Run a single test case."""
        self.init_cuda()
        self.setup()
        custom_kernel = self.get_custom_kernel()
        generate_input = self.get_generate_input()
        check_implementation = self.get_check_implementation()

        data = generate_input(**test.args)
        torch.cuda.synchronize()
        try:
            output = custom_kernel(clone_data(data))
        except Exception as E:
            return self.handle_kernel_error(E)
        torch.cuda.synchronize()
        return check_implementation(data, output)

    def run_single_benchmark(
        self, test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float
    ) -> BenchmarkResult:
        """Run benchmark - single iteration mode."""
        self.init_cuda()
        self.setup()
        custom_kernel = self.get_custom_kernel()
        generate_input = self.get_generate_input()
        check_implementation = self.get_check_implementation()

        durations = []
        correctness_error = None
        data = generate_input(**test.args)
        check_copy = clone_data(data)

        # Ensure kernel is compiled
        try:
            self.call_compile_kernel()
            torch.cuda.synchronize()
        except Exception as E:
            return BenchmarkResult(stats=None, error=f"Compilation failed: {E}")

        # Correctness check
        try:
            output = custom_kernel(clone_data(data))
        except Exception as E:
            return BenchmarkResult(stats=None, error=f"Encountered {E}")
        good, message = check_implementation(check_copy, output)
        if not good:
            correctness_error = message

        # Timing runs
        bm_start_time = time.perf_counter_ns()
        for i in range(max_repeats):
            if recheck and "seed" in test.args:
                test.args["seed"] += 13
                data = generate_input(**test.args)
                check_copy = clone_data(data)

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            self._clear_cache()

            start_event.record()
            output = custom_kernel(data)
            end_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(end_event) * 1e6  # ms to ns

            if recheck:
                good, message = check_implementation(check_copy, output)
                if not good and correctness_error is None:
                    correctness_error = message

            del output
            durations.append(duration)

            total_bm_duration = time.perf_counter_ns() - bm_start_time
            if i > 1 and total_bm_duration > 1e8:  # at least 2 runs and 100ms total
                stats = calculate_stats(durations)
                if (stats.err / stats.mean < 0.001 or
                    stats.mean * stats.runs > max_time_ns or
                    total_bm_duration > 120e9):
                    break

        return BenchmarkResult(stats=calculate_stats(durations), error=correctness_error)

    def run_single_benchmark_batched(
        self, test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float
    ) -> BenchmarkResult:
        """Run benchmark - batched iteration mode."""
        self.init_cuda()
        self.setup()
        custom_kernel = self.get_custom_kernel()
        generate_input = self.get_generate_input()
        check_implementation = self.get_check_implementation()

        durations = []
        data_list = []
        correctness_error = None

        # Generate batch of inputs
        for i in range(self.batch_size):
            if "seed" in test.args:
                test.args["seed"] += 42
            data = generate_input(**test.args)
            data_list.append(data)

        check_copy = clone_data(data_list)

        # Correctness check
        outputs = []
        try:
            for data in data_list:
                output = custom_kernel(clone_data(data))
                outputs.append(output)
        except Exception as E:
            return BenchmarkResult(stats=None, error=f"Encountered {E}")

        for ref_output, cust_output in zip(check_copy, outputs):
            good, message = check_implementation(ref_output, cust_output)
            if not good:
                correctness_error = message
                break

        # Timing runs
        bm_start_time = time.perf_counter_ns()
        for i in range(max_repeats):
            torch.cuda.synchronize()
            outputs = []
            self._clear_cache()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for data in data_list:
                output = custom_kernel(data)
                outputs.append(output)
            end_event.record()
            torch.cuda.synchronize()
            duration = (start_event.elapsed_time(end_event) / self.batch_size) * 1e6

            if recheck:
                for ref_output, cust_output in zip(check_copy, outputs):
                    good, message = check_implementation(ref_output, cust_output)
                    if not good and correctness_error is None:
                        correctness_error = message
                        break

            durations.append(duration)

            total_bm_duration = time.perf_counter_ns() - bm_start_time
            if i > 1 and total_bm_duration > 1e8:
                stats = calculate_stats(durations)
                if (stats.err / stats.mean < 0.001 or
                    stats.mean * stats.runs > max_time_ns or
                    total_bm_duration > 120e9):
                    break

        return BenchmarkResult(stats=calculate_stats(durations), error=correctness_error)

    def run_single_profile(self, test: TestCase) -> str:
        """Run profiling."""
        from torch.profiler import profile, ProfilerActivity

        self.init_cuda()
        self.setup()
        custom_kernel = self.get_custom_kernel()
        generate_input = self.get_generate_input()

        data = generate_input(**test.args)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            custom_kernel(clone_data(data))
            torch.cuda.synchronize()

        return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)


def _make_test_runner(runner: EvalRunner, test: TestCase):
    return runner.run_single_test(test)


def _make_benchmark_runner(runner: EvalRunner, test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float):
    if runner.use_batched_benchmark:
        return runner.run_single_benchmark_batched(test, recheck, max_repeats, max_time_ns)
    return runner.run_single_benchmark(test, recheck, max_repeats, max_time_ns)


def _make_compile_runner(runner: EvalRunner):
    return runner.compile_kernel_once()


def _make_profile_runner(runner: EvalRunner, test: TestCase):
    return runner.run_single_profile(test)


def run_testing(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase], runner: EvalRunner):
    """Run testing mode."""
    if runner.use_cutlass:
        logger.log("compile", "start")
        compile_success, compile_error = pool.apply(_make_compile_runner, (runner,))
        if not compile_success:
            logger.log("compile", "fail")
            logger.log("compile.error", compile_error)
            return 112
        logger.log("compile", "pass")

    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"test.{idx}.spec", test.spec)
        good, message = pool.apply(_make_test_runner, (runner, test))
        if not good:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", message)
            passed = False
        else:
            logger.log(f"test.{idx}.status", "pass")
            if message:
                logger.log(f"test.{idx}.message", message)

    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def run_benchmarking(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase], runner: EvalRunner):
    """Run benchmarking mode."""
    if runner.use_cutlass:
        logger.log("compile", "start")
        compile_success, compile_error = pool.apply(_make_compile_runner, (runner,))
        if not compile_success:
            logger.log("compile", "fail")
            logger.log("compile.error", compile_error)
            return 112
        logger.log("compile", "pass")

    # Warmup
    pool.apply(_make_benchmark_runner, (runner, tests[0], False, 100, 10e7))

    passed = True
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        result = pool.apply(_make_benchmark_runner, (runner, test, False, 100, 10e9))
        if result.stats is not None:
            for field in dataclasses.fields(Stats):
                logger.log(f"benchmark.{idx}.{field.name}", getattr(result.stats, field.name))
        if result.error is not None:
            passed = False
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", result.error)
        elif result.stats is None:
            passed = False
            logger.log(f"benchmark.{idx}.status", "fail")

    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def run_leaderboard(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase], runner: EvalRunner):
    """Run leaderboard mode."""
    if runner.use_cutlass:
        logger.log("compile", "start")
        compile_success, compile_error = pool.apply(_make_compile_runner, (runner,))
        if not compile_success:
            logger.log("compile", "fail")
            logger.log("compile.error", compile_error)
            return 112
        logger.log("compile", "pass")

    # Warmup all test shapes to ensure consistent benchmarking
    for test in tests:
        pool.apply(_make_benchmark_runner, (runner, test, False, 50, 5e8))

    logger.log("benchmark-count", len(tests))
    passed = True
    for i, test in enumerate(tests):
        result = pool.apply(_make_benchmark_runner, (runner, test, True, 100, 30e9))
        logger.log(f"benchmark.{i}.spec", test.spec)
        if result.stats is not None:
            for field in dataclasses.fields(Stats):
                logger.log(f"benchmark.{i}.{field.name}", getattr(result.stats, field.name))
        if result.error is not None:
            passed = False
            logger.log(f"benchmark.{i}.status", "fail")
            logger.log(f"benchmark.{i}.error", result.error)
        elif result.stats is None:
            passed = False
            logger.log(f"benchmark.{i}.status", "fail")
            break

    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def run_profiling(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase], runner: EvalRunner):
    """Run profiling mode."""
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        report = pool.apply(_make_profile_runner, (runner, test))
        logger.log(
            f"benchmark.{idx}.report",
            base64.b64encode(report.encode("utf-8"), b"+*").decode("utf-8"),
        )
    logger.log("check", "pass")
    return 0


def main(runner: EvalRunner):
    """Main entry point for eval scripts."""
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return 111

    if len(sys.argv) < 3:
        return 2

    mode = sys.argv[1]
    seed = os.getenv("POPCORN_SEED")
    os.unsetenv("POPCORN_SEED")
    seed = int(seed) if seed else None
    set_seed(seed or 42)

    tests = get_test_cases(sys.argv[2], seed)

    with PopcornOutput(int(fd)) as logger:
        mp_context = multiprocessing.get_context("spawn")
        with mp_context.Pool(1) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests, runner)
            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests, runner)
            if mode == "leaderboard":
                return run_leaderboard(logger, pool, tests, runner)
            if mode == "profile":
                return run_profiling(logger, pool, tests, runner)
            return 2
