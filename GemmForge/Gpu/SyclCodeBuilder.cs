using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class SyclCodeBuilder : IGPUCodeBuilder
    {
        private readonly Code _code;
        private readonly IExpressionResolver _expressionResolver;

        public SyclCodeBuilder(Code code)
        {
            _code = code;
            _expressionResolver = new CppExpressionResolver();
        }

        public IGPUCodeBuilder MallocSharedMemory(Malloc malloc)
        {
            malloc.CountExpression.Resolve(_expressionResolver);
            var expString = _expressionResolver.ExtractResult();
            
            _code.Append($"{malloc.Variable.TypeString} *{malloc.Variable.VariableName} = malloc_shared<{malloc.Variable.TypeString}>({expString});\n");
            return this;
        }

        public IGPUCodeBuilder DeclareKernelRange(Range localCount, Range localSize)
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

            _code.Append(s1 + s2);
            return this;
        }

        public IGPUCodeBuilder InitStreamByPointer(Stream stream, Variable ptr)
        {
            _code.Append($"queue {stream.Name} = static_cast<queue>({ptr.VariableName});\n");
            return this;
        }

        public IGPUCodeBuilder LaunchKernel(KernelFunction function)
        {
            var comma = function.Args.Size > 0 ? ", " : string.Empty;
            _code.Append($"{function.Name}({function.Args.Concat()}{comma}{function.Block.Name}, {function.Grid.Name}, {function.Stream.Name});\n");
            return this;
        }

        public Code Build()
        {
            return _code;
        }

        public IGPUCodeBuilder DefineKernel(KernelFunction func)
        {
            //ToDo: multiply range to global range
            var retType = "void";
            var body = func.Builder.Build();
            var args = func.Args.Concat();
            var kernelBody = $"{func.Stream.Name}->submit([&](handler &cgh){{" +
                             $"cgh.parallel_for(nd_range<3>{{{func.Block.Name}, {func.Grid.Name}}}, [=](nd_item<3> item){{\n" +
                             body +
                             "});\n" +
                             "});";

            var comma = func.Args.Size > 0 ? ", " : string.Empty;
            var text = $"{retType} {func.Name}({args}{comma}range<3> {func.Block.Name}, range<3> {func.Grid.Name}, queue *{func.Stream.Name}){{\n{kernelBody}}}\n";
            _code.Append(text);
            return this;
        }
    }
}