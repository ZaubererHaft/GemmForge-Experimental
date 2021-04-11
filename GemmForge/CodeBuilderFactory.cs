using GemmForge.Gpu;
using GemmForge.Host;

namespace GemmForge
{
    public class CodeBuilderFactory
    {
        public CodeBuilder CreateCppSyclCodeBuilder()
        {
            var code = new Code();
            var hostGenerator = new CppHostCodeGenerator(code);
            var gpuGenerator = new SyclCodeGenerator(code);
            return new CodeBuilder(hostGenerator, gpuGenerator, code);
        }
    }
}