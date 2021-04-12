using GemmForge.Gpu;

namespace GemmForge
{
    public class CodeBuilder
    {
        public CodeBuilder(Code code, HostCodeBuilder hostCodeBuilder, IGPUCodeBuilder gpuCodeBuilder)
        {
            Code = code;
            HostCodeBuilder = hostCodeBuilder;
            GpuCodeBuilder = gpuCodeBuilder;
        }

        public Code Code { get; }
        public HostCodeBuilder HostCodeBuilder { get; }
        public IGPUCodeBuilder GpuCodeBuilder { get; }

        public string Generate()
        {
            return Code.ToString();
        }
        
    }
}