using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class KernelFunction 
    {
        public string Name { get; }
        public FunctionArguments Args { get; }
        public IGPUCodeBuilder Builder { get; }
        public Range Block { get; }
        public Range Grid { get; }
        public Stream Stream { get; }

        public KernelFunction(string name, FunctionArguments args, IGPUCodeBuilder builder, Range block, Range grid, Stream stream)
        {
            Name = name;
            Args = args;
            Builder = builder;
            Block = block;
            Grid = grid;
            Stream = stream;
        }
    }
}