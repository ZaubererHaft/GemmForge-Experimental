using GemmForge.Common;
using GemmForge.Gpu;
using GemmForge.Host;

namespace GemmForge
{
    public class CodeBuilder
    {
        private readonly IHostCodeGenerator _hostCodeGenerator;
        private readonly IGPUCodeGenerator _gpuCodeGenerator;
        private readonly Code _code;

        public CodeBuilder(IHostCodeGenerator hostCodeGenerator, IGPUCodeGenerator gpuCodeGenerator, Code code)
        {
            _hostCodeGenerator = hostCodeGenerator;
            _gpuCodeGenerator = gpuCodeGenerator;
            _code = code;
        }

        public Code Build()
        {
            return _code;
        }

        public CodeBuilder Declare(Variable variable, Assignment init)
        {
            _hostCodeGenerator.DeclareAndAssign(variable, init);
            return this;
        }
        
        public CodeBuilder Declare(Pointer pointer, Assignment init)
        {
            _hostCodeGenerator.DeclareAndAssign(pointer, init);
            return this;
        }
        
    }
}