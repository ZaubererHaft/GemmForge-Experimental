namespace GemmForge.Gpu
{
    public interface IGPUCodeGenerator
    {
        string MallocSharedMemory(Malloc malloc);
    }
}