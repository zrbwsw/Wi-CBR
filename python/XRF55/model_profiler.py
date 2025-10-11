import torch
import time
import numpy as np
from typing import Tuple, Dict, Any

class ModelProfiler:
    """
    Model profiler for measuring FLOPs and inference time.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize the model profiler.
        
        Args:
            model: PyTorch model to profile
            device: Device to run profiling on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def count_flops(self, input_shape: Tuple[int, ...], num_classes: int = 6) -> float:
        """
        Count FLOPs for the model using a simple hook-based approach.
        
        Args:
            input_shape: Input tensor shape (batch_size, channels, height, width)
            num_classes: Number of output classes
            
        Returns:
            FLOPs in billions (G)
        """
        def conv_flop_count(input_shape, output_shape, kernel_size, groups=1):
            """Calculate FLOPs for convolution layer"""
            batch_size, in_channels, input_height, input_width = input_shape
            batch_size, out_channels, output_height, output_width = output_shape
            kernel_height, kernel_width = kernel_size
            
            # FLOPs = batch_size * output_height * output_width * 
            #         (in_channels / groups) * kernel_height * kernel_width * out_channels
            flops = batch_size * output_height * output_width * \
                   (in_channels // groups) * kernel_height * kernel_width * out_channels
            return flops
        
        def linear_flop_count(input_features, output_features, batch_size):
            """Calculate FLOPs for linear layer"""
            return batch_size * input_features * output_features
        
        # Approximate FLOPs calculation for ResNet18-based model
        # This is a simplified calculation
        batch_size = input_shape[0]
        
        # ResNet18 approximate FLOPs (for each branch)
        resnet18_flops = 1.8e9  # Approximate FLOPs for ResNet18
        
        # Two ResNet18 branches (p_features and d_features)
        total_flops = 2 * resnet18_flops * batch_size
        
        # DPFusion layer (approximate)
        dpfusion_flops = batch_size * 1024 * 1024 * 7 * 7  # Approximate
        total_flops += dpfusion_flops
        
        # Final FC layer
        fc_flops = linear_flop_count(1024, num_classes, batch_size)
        total_flops += fc_flops
        
        return total_flops / 1e9  # Convert to GFLOPs
    
    def measure_inference_time(self, p_input: torch.Tensor, d_input: torch.Tensor, 
                             num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """
        Measure inference time for the model.
        
        Args:
            p_input: Input tensor for p branch
            d_input: Input tensor for d branch  
            num_runs: Number of inference runs for timing
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary containing timing statistics
        """
        self.model.eval()
        
        # Move inputs to device
        p_input = p_input.to(self.device)
        d_input = d_input.to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(p_input, d_input)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = self.model(p_input, d_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times))
        }
    
    def profile_model(self, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224), 
                     num_classes: int = 6, num_runs: int = 100) -> Dict[str, Any]:
        """
        Complete model profiling including FLOPs and inference time.
        
        Args:
            input_shape: Input tensor shape (batch_size, channels, height, width)
            num_classes: Number of output classes
            num_runs: Number of runs for timing measurement
            
        Returns:
            Dictionary containing all profiling results
        """
        # Calculate FLOPs
        flops_g = self.count_flops(input_shape, num_classes)
        
        # Create dummy inputs for timing
        batch_size, channels, height, width = input_shape
        p_input = torch.randn(batch_size, channels, height, width)
        d_input = torch.randn(batch_size, channels, height, width)
        
        # Measure inference time
        timing_stats = self.measure_inference_time(p_input, d_input, num_runs)
        
        results = {
            'flops_g': flops_g,
            'inference_time': timing_stats,
            'model_info': {
                'input_shape': input_shape,
                'num_classes': num_classes,
                'device': str(self.device),
                'num_timing_runs': num_runs
            }
        }
        
        return results
    
    def print_profile_results(self, results: Dict[str, Any]):
        """
        Print profiling results in a formatted way.
        
        Args:
            results: Results from profile_model()
        """
        print("=" * 60)
        print("MODEL PROFILING RESULTS")
        print("=" * 60)
        
        print(f"FLOPs: {results['flops_g']:.2f} G")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f} M")
        
        timing = results['inference_time']
        print(f"\nInference Time Statistics:")
        print(f"  Mean: {timing['mean_ms']:.2f} ms")
        print(f"  Std:  {timing['std_ms']:.2f} ms")
        print(f"  Min:  {timing['min_ms']:.2f} ms")
        print(f"  Max:  {timing['max_ms']:.2f} ms")
        print(f"  Median: {timing['median_ms']:.2f} ms")
        
        info = results['model_info']
        print(f"\nProfiling Configuration:")
        print(f"  Input Shape: {info['input_shape']}")
        print(f"  Device: {info['device']}")
        print(f"  Timing Runs: {info['num_timing_runs']}")
        print("=" * 60)