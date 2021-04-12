using GemmForge.Common;
using GemmForge.Gpu;

namespace GemmForge
{
    public class CodeBuilder
    {
        private readonly IGPUCodeGenerator _gpuCodeGenerator;
        private readonly IExpressionResolver _expressionResolver;
        private readonly IVariableResolver _typeResolver;
        private readonly Code _code;

        public CodeBuilder(IGPUCodeGenerator gpuCodeGenerator)
        {
            _gpuCodeGenerator = gpuCodeGenerator;
            _expressionResolver = new CppExpressionResolver();
            _typeResolver = new CppVariableResolver();
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
            
            _code.AppendAndClose($"{typeString} {variable.VariableName} {assignmentExpression}");
            return this;
        }
        
        public CodeBuilder DeclarePointer(Variable variable, Assignment assignment)
        {
            variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.AppendAndClose($"{typeString} *{variable.VariableName} {assignmentExpression}");
            return this;
        }
        
        public CodeBuilder DeclarePointer(Variable variable)
        {
            variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            _code.AppendAndClose($"{typeString} *{variable.VariableName}");
            return this;
        }
        
        public CodeBuilder DeclareArray(Variable variable, Expression assignment)
        {
            variable.VariableType.Resolve(_typeResolver);
            var typeString = _typeResolver.ExtractResult();
            
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.AppendAndClose($"{typeString} {variable.VariableName}[{assignmentExpression}]");
            return this;
        }

        public CodeBuilder MallocGpuSharedMemory(Malloc expr)
        {
            var text = _gpuCodeGenerator.MallocSharedMemory(expr);
            _code.Append(text);
            return this;
        }
    }
}