using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class CudaCodeGenerator : IGPUCodeGenerator
    {
        private readonly IVariableResolver _typeConverter;
        private readonly IExpressionResolver _expressionResolver;

        public CudaCodeGenerator()
        {
            _typeConverter = new CppVariableResolver(this);
            _expressionResolver = new CppExpressionResolver(this);
        }
        public string Create(MallocShared assignment)
        {
            return string.Empty;
        }

        public string Create(SharedVariableType variable)
        {
            return $"__shared__ {variable.Type}";
        }
    }
}