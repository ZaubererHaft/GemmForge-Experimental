using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class SyclCodeGenerator : IGPUCodeGenerator
    {
        private readonly CppVariableTypeConverter _typeConverter;

        public SyclCodeGenerator()
        {
            _typeConverter = new CppVariableTypeConverter();
        }
        public string Resolve(MallocShared assignment)
        {
            return $"malloc_device<{_typeConverter.Convert(assignment.VariableType)}>({assignment.Count})";
        }
    }
}