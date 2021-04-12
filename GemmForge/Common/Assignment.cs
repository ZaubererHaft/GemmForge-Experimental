namespace GemmForge.Common
{
    public class Assignment : Expression
    {
        public Assignment(TypedExpression expression)
        {
            Expression = expression;
        }

        public TypedExpression Expression { get; }
        
        public override void Resolve(IExpressionResolver resolver)
        {
            resolver.Resolve(this);
        }
    }
}