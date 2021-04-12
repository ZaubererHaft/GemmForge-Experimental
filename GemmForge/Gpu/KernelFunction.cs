using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class KernelFunction : Function
    {
        public Range Block { get; }
        public Range Grid { get; }
        public Stream Stream { get; }

        public KernelFunction(string name, FunctionArguments args, KernelBuilder builder, Range block, Range grid, Stream stream) : base(name, new VoidType(), args, builder)
        {
            Block = block;
            Grid = grid;
            Stream = stream;
        }
    }
}