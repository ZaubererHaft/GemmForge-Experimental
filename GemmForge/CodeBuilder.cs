using GemmForge.Common;
using GemmForge.Gpu;

namespace GemmForge
{
    public class CodeBuilder
    {
        private readonly IExpressionResolver _expressionResolver;
        private readonly IVariableResolver _typeResolver;
        private readonly Code _code;

        public CodeBuilder(IGPUCodeGenerator gpuCodeGenerator)
        {
            _expressionResolver = new CppExpressionResolver(gpuCodeGenerator);
            _typeResolver = new CppVariableResolver(gpuCodeGenerator);
            _code = new Code();
        }

        public Code Build()
        {
            return _code;
        }

        public CodeBuilder DeclareVariable(Variable variable, Assignment assignment)
        {
            variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.Append($"{typeString} {variable.VariableName} {assignmentExpression}");
            return this;
        }
        
        public CodeBuilder DeclarePointer(Variable variable, Assignment assignment)
        {
            variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.Append($"{typeString} *{variable.VariableName} {assignmentExpression}");
            return this;
        }
        
        public CodeBuilder DeclarePointer(Variable variable)
        {
            variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            _code.Append($"{typeString} *{variable.VariableName}");
            return this;
        }
        
        public CodeBuilder DeclareArray(Variable variable, Expression assignment)
        {
            variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.Append($"{typeString} {variable.VariableName}[{assignmentExpression}]");
            return this;
        }

        public CodeBuilder MallocMemory(Variable variable, MallocShared expr)
        {
            variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            expr.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.Append($"{typeString} *{variable.VariableName}");
            _code.Append(assignmentExpression);
            
            return this;
        }
    }
}