using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class CudaCodeGenerator : IGPUCodeGenerator
    {
        private readonly IVariableResolver _typeResolver;
        private readonly IExpressionResolver _expressionResolver;

        public CudaCodeGenerator()
        {
            _typeResolver = new CppVariableResolver();
            _expressionResolver = new CppExpressionResolver();
        }
        
        public string MallocSharedMemory(Malloc malloc)
        {
            if (malloc.Hints == MallocHints.MallocLocal)
            {
                return LocalShareMemory(malloc);
            }
            
            malloc.Variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            malloc.CountExpression.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();

            var text = $"{typeString} *{malloc.Variable.VariableName};\n";
            text += $"cudaMallocManaged(&{malloc.Variable.VariableName}, {assignmentExpression} * sizeof({typeString}), cudaMemAttachGlobal);\n";
            return text;
        }

        private string LocalShareMemory(Malloc malloc)
        {
            malloc.Variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            malloc.CountExpression.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();

            return $"__shared__ {typeString} {malloc.Variable.VariableName}[{assignmentExpression}];\n";
        }
    }
}