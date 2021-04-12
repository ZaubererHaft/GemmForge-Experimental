using GemmForge.Common;
using GemmForge.Gpu;

namespace GemmForge
{
    public class HostCodeBuilder
    {
        private readonly IExpressionResolver _expressionResolver;
        private readonly Code _code;

        public HostCodeBuilder(Code code)
        {
            _code = code;
            _expressionResolver = new CppExpressionResolver();
        }

        public Code Build()
        {
            return _code;
        }

        public HostCodeBuilder DeclareVariable(Variable variable, Assignment assignment)
        {
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.AppendAndClose($"{variable.TypeString} {variable.VariableName} {assignmentExpression}");
            return this;
        }
        
        public HostCodeBuilder DeclarePointer(Variable variable, Assignment assignment)
        {
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.AppendAndClose($"{variable.TypeString} *{variable.VariableName} {assignmentExpression}");
            return this;
        }
        
        public HostCodeBuilder DeclarePointer(Variable variable)
        {
            _code.AppendAndClose($"{variable.VariableType.Type} *{variable.VariableName}");
            return this;
        }
        
        public HostCodeBuilder DeclareArray(Variable variable, Expression assignment)
        {
            assignment.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();
            
            _code.AppendAndClose($"{variable.TypeString} {variable.VariableName}[{assignmentExpression}]");
            return this;
        }

        public HostCodeBuilder DefineFunction(Function f)
        {
            var retType = f.ReturnType.Type;
            var body = f.BodyBuilder.Build();
            var args = f.FunctionArgs.Concat();

            var text = $"{retType} {f.Name}({args}){{\n{body}}}\n";
            
            _code.Append(text);
            return this;
        }

        public HostCodeBuilder Return(Expression expression)
        {
            expression.Resolve(_expressionResolver);
            var text = _expressionResolver.ExtractResult();
            _code.AppendAndClose($"return {text}");
            return this;
        }

    }
}