using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class SyclCodeGenerator : IGPUCodeGenerator
    {
        private readonly IVariableResolver _typeConverter;
        private readonly IExpressionResolver _expressionResolver;

        public SyclCodeGenerator()
        {
            _typeConverter = new CppVariableResolver(this);
            _expressionResolver = new CppExpressionResolver(this);
        }

        public string Create(MallocShared assignment)
        {
            assignment.Variable.VariableType.Resolve(_typeConverter);
            var typeString = _typeConverter.ExtractResult();

            assignment.Count.Resolve(_expressionResolver);
            var expString = _expressionResolver.ExtractResult();
            
            return $"{assignment.Variable.VariableName} = malloc_device<{typeString}>({expString})";
        }

        public string Create(SharedVariableType assignment)
        {
            return assignment.Type;
        }
    }
}