using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class SyclCodeGenerator : IGPUCodeGenerator
    {
        private readonly IVariableResolver _typeConverter;

        public SyclCodeGenerator()
        {
            _typeConverter = new CppVariableResolver(this);
        }
        public string Create(MallocShared assignment)
        {
            assignment.VariableType.Resolve(_typeConverter);
            var typeString = _typeConverter.ExtractResult();
            
            return $"malloc_device<{typeString}>({assignment.Count})";
        }

        public string Create(SharedVariableType assignment)
        {
            return assignment.Type;
        }
    }
}