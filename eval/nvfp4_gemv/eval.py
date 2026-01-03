import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.cuda
from common.eval_base import EvalRunner, main


class GemvEvalRunner(EvalRunner):
    use_cutlass = False
    use_batched_benchmark = True
    batch_size = 50
    use_large_cache_clear = True

    def get_custom_kernel(self):
        from submission import custom_kernel
        return custom_kernel

    def get_generate_input(self):
        from reference import generate_input
        return generate_input

    def get_check_implementation(self):
        from reference import check_implementation
        return check_implementation


if __name__ == "__main__":
    sys.exit(main(GemvEvalRunner()))
