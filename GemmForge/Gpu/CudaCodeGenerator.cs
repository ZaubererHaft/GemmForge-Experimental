using System;
using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class CudaCodeGenerator : IGPUCodeGenerator
    {
        private readonly IExpressionResolver _expressionResolver;

        public CudaCodeGenerator()
        {
            _expressionResolver = new CppExpressionResolver();
        }
        
        public string MallocSharedMemory(Malloc malloc)
        {
            if (malloc.Hints == MallocHints.MallocLocal)
            {
                return LocalShareMemory(malloc);
            }

            malloc.CountExpression.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();

            var text = $"{malloc.Variable.TypeString} *{malloc.Variable.VariableName};\n";
            text += $"cudaMallocManaged(&{malloc.Variable.VariableName}, {assignmentExpression} * sizeof({malloc.Variable.TypeString}), cudaMemAttachGlobal);\n";
            return text;
        }

        public string DeclareKernelRange(Range localCount, Range localSize)
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

            return s1 + s2;
        }

        public string InitStreamByPointer(Stream stream, Variable ptr)
        {
            return $"cudaStream_t {stream.Name} = static_cast<cudaStream_t>({ptr.VariableName});\n";
        }

        public string LaunchKernel(Range block, Range grid, Stream stream)
        {
            return string.Empty;
        }

        private string LocalShareMemory(Malloc malloc)
        {
            malloc.CountExpression.Resolve(_expressionResolver);
            var assignmentExpression = _expressionResolver.ExtractResult();

            return $"__shared__ {malloc.Variable.TypeString} {malloc.Variable.VariableName}[{assignmentExpression}];\n";
        }
    }
}