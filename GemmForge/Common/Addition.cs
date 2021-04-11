namespace GemmForge.Common
{
    public class Addition : BinaryExpression
    {
        public Addition(Expression expressionA, Expression expressionB) : base(expressionA, expressionB)
        {
        }

        public override void Resolve(IExpressionResolver resolver)
        {
            resolver.Resolve(this);
        }
    }
}