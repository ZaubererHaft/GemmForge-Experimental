using GemmForge.Gpu;

namespace GemmForge.Common
{
    public interface IExpressionResolver
    {
        void Resolve(Literal expr);
        void Resolve(Assignment expr);
        void Resolve(Addition expr);
        string ExtractResult();
    }
}