using GemmForge.Common;
using GemmForge.Gpu;

namespace GemmForge
{
    public class CodeBuilder
    {
        private readonly IExpressionResolver _expressionResolver;
        private readonly Code _code;
        private readonly CppVariableTypeConverter _typeConverter;

        public CodeBuilder(IGPUCodeGenerator gpuCodeGenerator)
        {
            _expressionResolver = new CppExpressionResolver(gpuCodeGenerator);
            _typeConverter = new CppVariableTypeConverter();
            _code = new Code();
        }

        public Code Build()
        {
            return _code;
        }

        public CodeBuilder DeclareVariable(Variable variable, Assignment assignment)
        {
            assignment.Resolve(_expressionResolver);            
            _code.Append($"{_typeConverter.Convert(variable.VariableType)} {variable.VariableName} {_expressionResolver.ExtractResult()}");
            return this;
        }
        
        public CodeBuilder DeclarePointer(Variable pointer, Assignment assignment)
        {
            assignment.Resolve(_expressionResolver);            
            _code.Append($"{_typeConverter.Convert(pointer.VariableType)} *{pointer.VariableName} {_expressionResolver.ExtractResult()}");
            return this;
        }
        
        public CodeBuilder DeclareArray(Variable pointer, Expression assignment)
        {
            assignment.Resolve(_expressionResolver);            
            _code.Append($"{_typeConverter.Convert(pointer.VariableType)} {pointer.VariableName}[{_expressionResolver.ExtractResult()}]");
            return this;
        }
 
    }
}