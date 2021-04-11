namespace GemmForge.Common
{
    public class Assignment : Expression
    {
        public Assignment(Expression expression)
        {
            Expression = expression;
        }

        public Expression Expression { get; }
        
        public override void Resolve(IExpressionResolver resolver)
        {
            resolver.Resolve(this);
        }
    }
}