using GemmForge.Common;

namespace GemmForge.Host
{
    public class CppHostCodeGenerator : IHostCodeGenerator
    {
        private readonly Code _code;
        private readonly CppExpressionResolver _expressionResolver;
        private readonly CppVariableTypeConverter _typeConverter;
        public CppHostCodeGenerator(Code code)
        {
            _code = code;
            _expressionResolver = new CppExpressionResolver();
            _typeConverter = new CppVariableTypeConverter();
        }

        public void DeclareAndAssign(Variable variable, Assignment assignment)
        {
            assignment.Resolve(_expressionResolver);            
            _code.Append($"{_typeConverter.Convert(variable.VariableType)} {variable.VariableName} {_expressionResolver.ResolvedString}");
            _expressionResolver.ResolvedString.Clear();
        }

        public void DeclareAndAssign(Pointer pointer, Assignment assignment)
        {
            assignment.Resolve(_expressionResolver);            
            _code.Append($"{_typeConverter.Convert(pointer.VariableType)} *{pointer.VariableName} {_expressionResolver.ResolvedString}");
            _expressionResolver.ResolvedString.Clear();
        }
    }
}