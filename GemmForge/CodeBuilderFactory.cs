using GemmForge.Gpu;

namespace GemmForge
{
    public class CodeBuilderFactory
    {
        public CodeBuilder CreateCppSyclCodeBuilder()
        {
            var gpuGenerator = new SyclCodeGenerator();
            return new CodeBuilder(gpuGenerator);
        }

        public CodeBuilder CreateCppCUDACodeBuilder()
        {
            var gpuGenerator = new CudaCodeGenerator();
            return new CodeBuilder(gpuGenerator);
        }
    }
}