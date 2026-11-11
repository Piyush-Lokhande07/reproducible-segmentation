import os
import random
import sys
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Enforce reproducibility across random generators and CUDA backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Enforce deterministic convolution algorithms in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_environment_info(output_file: str = "logs/environment_info.txt") -> None:
    """Records exact runtime versions required by the submission contract."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"GPU Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"cuDNN Version: {torch.backends.cudnn.version()}\n")
    print(f"Environment metadata written to {output_file}")
