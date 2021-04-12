namespace GemmForge.Common
{
    public class Variable : Expression
    {
        public VariableType VariableType { get; }
        public string VariableName { get; }

        public Variable(VariableType variableType, string variableName)
        {
            VariableType = variableType;
            VariableName = variableName;
        }

        public string TypeString => VariableType.Type;
        public override void Resolve(IExpressionResolver resolver)
        {
            resolver.Resolve(this);
        }
    }
}