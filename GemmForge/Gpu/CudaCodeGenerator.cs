using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class CudaCodeGenerator : IGPUCodeGenerator
    {
        private readonly IVariableResolver _typeConverter;

        public CudaCodeGenerator()
        {
            _typeConverter = new CppVariableResolver(this);
        }
        public string Create(MallocShared assignment)
        {
            return string.Empty;
        }

        public string Create(SharedVariableType assignment)
        {
            return $"__shared__ {assignment.Type}";
        }
    }
}