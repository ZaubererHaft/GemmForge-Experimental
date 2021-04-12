using GemmForge.Common;
using GemmForge.Gpu;

namespace GemmForge
{
    public class CodeBuilder
    {
        private readonly IGPUCodeGenerator _gpuCodeGenerator;
        private readonly IExpressionResolver _expressionResolver;
        private readonly Code _code;

        public CodeBuilder(IGPUCodeGenerator gpuCodeGenerator)
        {
            _gpuCodeGenerator = gpuCodeGenerator;
            _expressionResolver = new CppExpressionResolver();
            _code = new Code();
        }

        public Code Build()
        {
            return _code;
        }

        public CodeBuilder DeclareVariable(Variable variable, Assignment assignment)
        {
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.AppendAndClose($"{variable.TypeString} {variable.VariableName} {assignmentExpression}");
            return this;
        }
        
        public CodeBuilder DeclarePointer(Variable variable, Assignment assignment)
        {
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.AppendAndClose($"{variable.TypeString} *{variable.VariableName} {assignmentExpression}");
            return this;
        }
        
        public CodeBuilder DeclarePointer(Variable variable)
        {
            _code.AppendAndClose($"{variable.VariableType.Type} *{variable.VariableName}");
            return this;
        }
        
        public CodeBuilder DeclareArray(Variable variable, Expression assignment)
        {
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.AppendAndClose($"{variable.TypeString} {variable.VariableName}[{assignmentExpression}]");
            return this;
        }

        public CodeBuilder MallocGpuSharedMemory(Malloc expr)
        {
            var text = _gpuCodeGenerator.MallocSharedMemory(expr);
            _code.Append(text);
            return this;
        }
        
        public CodeBuilder DeclareGpuKernelRange(Range localCount, Range localSize)
        {
            var text = _gpuCodeGenerator.DeclareKernelRange(localCount, localSize);
            _code.Append(text);
            return this;
        }
        
    }
}