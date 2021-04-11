namespace GemmForge.Common
{
    public abstract class BinaryExpression : Expression
    {
        public BinaryExpression(Expression expressionA, Expression expressionB)
        {
            ExpressionA = expressionA;
            ExpressionB = expressionB;
        }

        public Expression ExpressionA { get; }
        public Expression ExpressionB { get; }


    }
}