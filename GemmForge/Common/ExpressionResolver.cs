using GemmForge.Gpu;

namespace GemmForge.Common
{
    public interface IExpressionResolver
    {
        void Resolve(Literal literal);
        void Resolve(Assignment assignment);
        void Resolve(MallocShared assignment);
        string ExtractResult();
    }
}