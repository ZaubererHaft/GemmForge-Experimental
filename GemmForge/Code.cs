using System.Text;

namespace GemmForge
{
    public class Code
    {
        private readonly StringBuilder _stringBuilder;

        public Code()
        {
            _stringBuilder = new StringBuilder();
        }

        public void AppendAndClose(string code)
        {
            _stringBuilder.Append(code).Append(";\n");
        }
        
        public void Append(string code)
        {
            _stringBuilder.Append(code);
        }

        public override string ToString()
        {
            return _stringBuilder.ToString();
        }
    }
}