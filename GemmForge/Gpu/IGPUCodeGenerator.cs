using GemmForge.Common;

namespace GemmForge.Gpu
{
    public interface IGPUCodeGenerator
    {
        string MallocSharedMemory(Malloc malloc);
        string DeclareKernelRange(Range localCount, Range localSize);
    }
}