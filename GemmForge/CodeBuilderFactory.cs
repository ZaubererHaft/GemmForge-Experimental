using GemmForge.Gpu;

namespace GemmForge
{
    public class CodeBuilderFactory
    {
        public CodeBuilder CreateCppSyclCodeBuilder()
        {
            var code = new Code();
            var gpuGenerator = new SyclCodeBuilder(code);
            var hostGenerator = new HostCodeBuilder(code);
            return new CodeBuilder(hostGenerator, gpuGenerator);
        }

        public CodeBuilder CreateCppCUDACodeBuilder()
        {
            var code = new Code();
            var gpuGenerator = new CudaCodeBuilder(code);
            var hostGenerator = new HostCodeBuilder(code);
            return new CodeBuilder(hostGenerator, gpuGenerator);
        }
    }
}