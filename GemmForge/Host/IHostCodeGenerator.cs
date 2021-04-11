using GemmForge.Common;

namespace GemmForge.Host
{
    public interface IHostCodeGenerator
    {
        void DeclareAndAssign(Variable hostVariable, Assignment init);
        void DeclareAndAssign(Pointer pointer, Assignment init);
    }
}