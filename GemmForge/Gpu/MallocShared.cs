using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class MallocShared : Expression
    {
        public VariableType VariableType { get; }
        public uint Count { get; }

        public MallocShared(VariableType variableType, uint count)
        {
            VariableType = variableType;
            Count = count;
        }

        public override void Resolve(IExpressionResolver resolver)
        {
            resolver.Resolve(this);
        }
    }
}