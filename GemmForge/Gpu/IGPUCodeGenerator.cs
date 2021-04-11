namespace GemmForge.Gpu
{
    public interface IGPUCodeGenerator
    {
        string Create(MallocShared assignment);
        string Create(SharedVariableType assignment);

    }
}