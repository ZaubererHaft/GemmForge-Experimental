using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class SyclCodeGenerator : IGPUCodeGenerator
    {
        private readonly IVariableResolver _typeConverter;
        private readonly IExpressionResolver _expressionResolver;

        public SyclCodeGenerator()
        {
            _typeConverter = new CppVariableResolver();
            _expressionResolver = new CppExpressionResolver();
        }

        public string MallocSharedMemory(Malloc malloc)
        {
            malloc.Variable.VariableType.Resolve(_typeConverter);
            var typeString = _typeConverter.ExtractResult();

            malloc.CountExpression.Resolve(_expressionResolver);
            var expString = _expressionResolver.ExtractResult();
            
            return $"{typeString} *{malloc.Variable.VariableName} = malloc_shared<{typeString}>({expString});\n";
        }
    }
}