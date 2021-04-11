using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class CudaCodeGenerator : IGPUCodeGenerator
    {
        private readonly CppVariableTypeConverter _typeConverter;

        public CudaCodeGenerator()
        {
            _typeConverter = new CppVariableTypeConverter();
        }
        public string Resolve(MallocShared assignment)
        {
            return $"__shared__";
        }
    }
}