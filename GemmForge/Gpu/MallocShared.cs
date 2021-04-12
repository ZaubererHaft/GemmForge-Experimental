using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class MallocShared : Expression
    {
        public Expression Count { get; }
        public Variable Variable { get; }
        
        public MallocShared(Variable variable, Expression count)
        {
            Variable = variable;
            Count = count;
        }

        public override void Resolve(IExpressionResolver resolver)
        {
            resolver.Resolve(this);
        }
    }
}