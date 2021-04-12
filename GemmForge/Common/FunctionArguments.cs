namespace GemmForge.Common
{
    public class FunctionArguments
    {
        private readonly Variable[] _vars;

        public FunctionArguments(params Variable[] vars)
        {
            _vars = vars;
        }
        
        public string Concat()
        {
            var text = string.Empty;
            for (var i = 0; i < _vars.Length; i++)
            {
                text += _vars[i].TypeString + " " + _vars[i].VariableName;
                if (i < _vars.Length - 1)
                {
                    text += ", ";
                }
            }

            return text;
        }

        public int Size => _vars.Length;
    }
}