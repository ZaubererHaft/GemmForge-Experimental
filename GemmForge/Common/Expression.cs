namespace GemmForge.Common
{
    public abstract class Expression
    {
        public abstract void Resolve(IExpressionResolver resolver);
    }
}