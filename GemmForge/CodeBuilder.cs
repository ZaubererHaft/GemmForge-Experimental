using GemmForge.Gpu;

namespace GemmForge
{
    public class CodeBuilder
    {
        public CodeBuilder(HostCodeBuilder hostCodeBuilder, IGPUCodeBuilder gpuCodeBuilder)
        {
            HostCodeBuilder = hostCodeBuilder;
            GpuCodeBuilder = gpuCodeBuilder;
        }

        public HostCodeBuilder HostCodeBuilder { get; }
        public IGPUCodeBuilder GpuCodeBuilder { get; }
        
    }
}