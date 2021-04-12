using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class Range : GpuVariable
    {
        public Expression X { get; }
        public Expression Y { get; }
        public Expression Z { get; }

        public Range(string name, Expression x, Expression y, Expression z) : base(name)
        {
            X = x;
            Y = y;
            Z = z;
        }
    }
}