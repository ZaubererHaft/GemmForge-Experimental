using System.Text;
using GemmForge.Common;
using GemmForge.Gpu;

namespace GemmForge.Host
{
    public class CppExpressionResolver : IExpressionResolver
    {

        public StringBuilder ResolvedString { get; } = new StringBuilder();

        public void Resolve(Assignment assignment)
        {
            ResolvedString.Append($"= ");
            assignment.Expression.Resolve(this);
            ResolvedString.Append($";\n");
        }

        public void Resolve(MallocShared assignment)
        {
            throw new System.NotImplementedException();
        }

        public void Resolve(Literal literal)
        {
            ResolvedString.Append($"{literal.Value}");
        }
    }
}