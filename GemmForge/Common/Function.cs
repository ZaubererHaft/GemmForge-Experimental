namespace GemmForge.Common
{
    public class Function
    {
        public Function(string name, VariableType returnType, FunctionArguments functionArgs, CodeBuilder bodyBuilder)
        {
            Name = name;
            ReturnType = returnType;
            FunctionArgs = functionArgs;
            BodyBuilder = bodyBuilder;
        }

        public string Name { get; }
        public VariableType ReturnType{ get; }
        public FunctionArguments FunctionArgs{ get; }
        public CodeBuilder BodyBuilder{ get; }
    }
}