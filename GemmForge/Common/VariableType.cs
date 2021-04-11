namespace GemmForge.Common
{
    public abstract class VariableType
    {
        public abstract void Resolve(IVariableResolver resolver);
        public abstract string Type { get; }
    }
}