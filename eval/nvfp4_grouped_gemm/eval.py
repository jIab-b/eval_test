import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.cuda
from common.eval_base import EvalRunner, main


class GroupedGemmEvalRunner(EvalRunner):
    use_cutlass = True

    def init_cuda(self):
        import cutlass
        torch.cuda.init()
        cutlass.cuda.initialize_cuda_context()

    def get_custom_kernel(self):
        from submission import custom_kernel
        return custom_kernel

    def get_generate_input(self):
        from reference import generate_input
        return generate_input

    def get_check_implementation(self):
        from reference import check_implementation
        return check_implementation

    def handle_kernel_error(self, e):
        from cutlass.cute.nvgpu.common import OpError
        if isinstance(e, OpError):
            print(f"Encountered {e}", file=sys.stderr)
            return False, str(e)
        return super().handle_kernel_error(e)


if __name__ == "__main__":
    sys.exit(main(GroupedGemmEvalRunner()))
