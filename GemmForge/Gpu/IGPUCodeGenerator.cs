namespace GemmForge.Gpu
{
    public interface IGPUCodeGenerator
    {
        string Resolve(MallocShared assignment);
    }
}