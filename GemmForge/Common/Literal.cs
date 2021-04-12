namespace GemmForge.Common
{
    public class Literal : TypedExpression
    {
        public string Value { get; }

        public Literal(string value)
        {
            Value = value;
        }

        public override void Resolve(IExpressionResolver resolver)
        {
            resolver.Resolve(this);
        }
    }
}