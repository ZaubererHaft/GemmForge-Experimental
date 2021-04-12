using GemmForge.Common;

namespace GemmForge.Gpu
{
    public interface IGPUCodeGenerator
    {
        string MallocSharedMemory(Malloc malloc);
        string DeclareKernelRange(Range localCount, Range localSize);
        string InitStreamByPointer(Stream stream, Variable ptr);
        string LaunchKernel(Range block, Range grid, Stream stream);
        string DefineKernel(KernelFunction func);
    }
}