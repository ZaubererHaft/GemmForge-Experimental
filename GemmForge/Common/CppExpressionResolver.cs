using System.Text;
using GemmForge.Gpu;

namespace GemmForge.Common
{
    public class CppExpressionResolver : IExpressionResolver
    {
        private readonly StringBuilder _textBuilder;
        private readonly IGPUCodeGenerator _gpuCodeGenerator;

        public CppExpressionResolver(IGPUCodeGenerator gpuCodeGenerator)
        {
            _gpuCodeGenerator = gpuCodeGenerator;
            _textBuilder = new StringBuilder();
        }

        public void Resolve(Assignment assignment)
        {
            _textBuilder.Append($"= ");
            assignment.Expression.Resolve(this);
        }

        public void Resolve(Addition addition)
        {
            addition.ExpressionA.Resolve(this);
            _textBuilder.Append(" + ");
            addition.ExpressionB.Resolve(this);   
        }

        public void Resolve(MallocShared assignment)
        {
            _textBuilder.Append(_gpuCodeGenerator.Create(assignment));
        }

        public void Resolve(Literal literal)
        {
            _textBuilder.Append($"{literal.Value}");
        }
        
        public string ExtractResult()
        {
            var text = _textBuilder.ToString();
            _textBuilder.Clear();
            return text;
        }
    }
}