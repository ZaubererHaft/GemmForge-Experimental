using System.Text;
using GemmForge.Gpu;

namespace GemmForge.Common
{
    public class CppVariableResolver : IVariableResolver
    {
        private readonly StringBuilder _textBuilder;

        public CppVariableResolver()
        {
            _textBuilder = new StringBuilder();
        }
        
        public void Resolve(SinglePrecisionFloat variableType)
        {
            _textBuilder.Append(variableType.Type);
        }

        public void Resolve(DoublePrecisionFloat variableType)
        {
            _textBuilder.Append(variableType.Type);
        }

        public string ExtractResult()
        {
            var text = _textBuilder.ToString();
            _textBuilder.Clear();
            return text;
        }
    }
}