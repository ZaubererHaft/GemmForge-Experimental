namespace GemmForge.Common
{
    public class Variable
    {
        public VariableType VariableType { get; }
        public string VariableName { get; }

        public Variable(VariableType variableType, string variableName)
        {
            VariableType = variableType;
            VariableName = variableName;
        }
    }
}