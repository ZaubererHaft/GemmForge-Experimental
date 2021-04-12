using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class KernelFunction : Function
    {
        public Range Block { get; }
        public Range Grid { get; }
        public Stream Stream { get; }

        public KernelFunction(Function function, Range block, Range grid, Stream stream) : base(function.Name, function.ReturnType, function.FunctionArgs, function.BodyBuilder)
        {
            Block = block;
            Grid = grid;
            Stream = stream;
        }
    }
}