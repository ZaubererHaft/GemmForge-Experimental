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
        public string Create(MallocShared exp)
        {
            return $"cudaMallocManaged(&TEST, {exp.Count}, cudaMemAttachGlobal)";
        }

        public string Create(SharedVariableType variable)
        {
            return $"__shared__ {variable.Type}";
        }
    }
}