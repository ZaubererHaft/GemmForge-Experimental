using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class Malloc 
    {
        public Expression CountExpression { get; }
        public MallocHints Hints { get; }
        public Variable Variable { get; }
        
        public Malloc(Variable variable, Expression expr, MallocHints hints = MallocHints.MallocGlobal)
        {
            Variable = variable;
            CountExpression = expr;
            Hints = hints;
        }
    }
}