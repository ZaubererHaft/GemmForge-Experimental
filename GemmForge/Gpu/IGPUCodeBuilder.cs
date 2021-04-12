using GemmForge.Common;

namespace GemmForge.Gpu
{
    public interface IGPUCodeBuilder
    {
        IGPUCodeBuilder MallocSharedMemory(Malloc malloc);
        IGPUCodeBuilder DeclareKernelRange(Range localCount, Range localSize);
        IGPUCodeBuilder InitStreamByPointer(Stream stream, Variable ptr);
        IGPUCodeBuilder DefineKernel(KernelFunction func);
        IGPUCodeBuilder LaunchKernel(KernelFunction function);
        Code Build();
    }
}