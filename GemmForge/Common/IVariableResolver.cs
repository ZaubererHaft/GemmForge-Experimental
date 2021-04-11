using GemmForge.Gpu;

namespace GemmForge.Common
{
    public interface IVariableResolver
    {
        void Resolve(SinglePrecisionFloat variableType);
        void Resolve(DoublePrecisionFloat variableType);
        void Resolve(SharedVariableType variableType);
        string ExtractResult();
    }
}