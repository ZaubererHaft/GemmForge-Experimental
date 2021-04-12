using System;
using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class CudaCodeBuilder : IGPUCodeBuilder
    {
        private readonly Code _code;
        private readonly IExpressionResolver _expressionResolver;

        public CudaCodeBuilder(Code code)
        {
            _code = code;
            _expressionResolver = new CppExpressionResolver();
        }
        
        public IGPUCodeBuilder MallocSharedMemory(Malloc malloc)
        {
            if (malloc.Hints == MallocHints.MallocLocal)
            {
                return LocalShareMemory(malloc);
            }

            malloc.CountExpression.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();

            var text = $"{malloc.Variable.TypeString} *{malloc.Variable.VariableName};\n";
            text += $"cudaMallocManaged(&{malloc.Variable.VariableName}, {assignmentExpression} * sizeof({malloc.Variable.TypeString}), cudaMemAttachGlobal)";
            
            _code.AppendAndClose(text);
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

            var s1 = "dim3 " + localCount.Name + " (" + countXExp + ", " + countYExp + ", " + countZExp + ");\n";
            
            localSize.X.Resolve(_expressionResolver);
            var sizeXExp = _expressionResolver.ExtractResult();
            localSize.Y.Resolve(_expressionResolver);
            var sizeYExp = _expressionResolver.ExtractResult();
            localSize.Z.Resolve(_expressionResolver);
            var sizeZExp = _expressionResolver.ExtractResult();
            
            var s2 = "dim3 " + localSize.Name + " (" + sizeXExp + ", " + sizeYExp + ", " + sizeZExp + ");\n";

            _code.Append(s1 + s2);
            return this;
        }

        public IGPUCodeBuilder InitStreamByPointer(Stream stream, Variable ptr)
        {
            _code.AppendAndClose($"cudaStream_t {stream.Name} = static_cast<cudaStream_t>({ptr.VariableName})");
            return this;
        }

        public IGPUCodeBuilder DefineKernel(KernelFunction func)
        {
            var body = func.Builder.Build();
            var args = func.Args.Concat();

            _code.Append($"__global__ __launch_bounds__(64) void {func.Name}({args}){{\n{body}}}\n");
            return this;
        }

        public IGPUCodeBuilder LaunchKernel(KernelFunction function)
        {
            _code.Append($"{function.Name}<<<{function.Grid.Name}, {function.Block.Name}, 0, {function.Stream.Name}>>>({function.Args.Concat()});\n");
            return this;
        }

        public Code Build()
        {
            return _code;
        }

        private IGPUCodeBuilder LocalShareMemory(Malloc malloc)
        {
            malloc.CountExpression.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();

            _code.Append($"__shared__ {malloc.Variable.TypeString} {malloc.Variable.VariableName}[{assignmentExpression}];\n");
            return this;
        }
    }
}