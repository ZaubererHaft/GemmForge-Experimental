using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class SyclCodeGenerator : IGPUCodeGenerator
    {
        private readonly IExpressionResolver _expressionResolver;

        public SyclCodeGenerator()
        {
            _expressionResolver = new CppExpressionResolver();
        }

        public string MallocSharedMemory(Malloc malloc)
        {
            malloc.CountExpression.Resolve(_expressionResolver);
            var expString = _expressionResolver.ExtractResult();
            
            return $"{malloc.Variable.TypeString} *{malloc.Variable.VariableName} = malloc_shared<{malloc.Variable.TypeString}>({expString});\n";
        }

        public string DeclareKernelRange(Range localCount, Range localSize)
        {
            localCount.X.Resolve(_expressionResolver);
            var countXExp = _expressionResolver.ExtractResult();
            localCount.Y.Resolve(_expressionResolver);
            var countYExp = _expressionResolver.ExtractResult();
            localCount.Z.Resolve(_expressionResolver);
            var countZExp = _expressionResolver.ExtractResult();

            var s1 = "range<3> " + localCount.Name + " {" + countXExp + ", " + countYExp + ", " + countZExp + "};\n";
            
            localSize.X.Resolve(_expressionResolver);
            var sizeXExp = _expressionResolver.ExtractResult();
            localSize.Y.Resolve(_expressionResolver);
            var sizeYExp = _expressionResolver.ExtractResult();
            localSize.Z.Resolve(_expressionResolver);
            var sizeZExp = _expressionResolver.ExtractResult();
            
            var s2 = "range<3> " + localSize.Name + " {" + sizeXExp + ", " + sizeYExp + ", " + sizeZExp + "};\n";

            return s1 + s2;
        }

        public string InitStreamByPointer(Stream stream, Variable ptr)
        {
            return $"queue {stream.Name} = static_cast<queue>({ptr.VariableName});\n";
        }

        public string LaunchKernel(Range block, Range grid, Stream stream)
        {
            return string.Empty;
        }

        public string DefineKernel(KernelFunction func)
        {
            var retType = func.ReturnType.Type;
            var body = func.BodyBuilder.Build();
            var args = func.FunctionArgs.Concat();
            var kernelBody = $"{func.Stream.Name}->submit([&](handler &cgh){{" +
                             $"cgh.parallel_for(nd_range<3>{{{func.Block.Name}, {func.Grid.Name}}}, [=](nd_item<3> item){{\n" +
                             body +
                             "});\n" +
                             "});";

            var comma = func.FunctionArgs.Size > 0 ? ", " : string.Empty;
            var text = $"{retType} {func.Name}({args}{comma}range<3> {func.Block.Name}, range<3> {func.Grid.Name}, queue *{func.Stream.Name}){{\n{kernelBody}}}";
            return text;
        }
    }
}